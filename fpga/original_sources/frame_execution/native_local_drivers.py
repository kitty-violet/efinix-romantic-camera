"""Replicate only measured public data drivers, with reversible graph proof.

No latency, state encoding, clock, reset, arithmetic or protected PHY changes.
Geometry is used to group loads, never to fix ordinary cells to coordinates.
"""
import argparse,collections,json,re,traceback
from pathlib import Path
from audit_snapshot_cells import cells
from audit_native_roundtrip import tokens
from audit_local_reset_implementation import norm
from probe_full_native_split_map import params
from complete_native_logic_split import protected_blocks
from run_units import ROOT,sha

CELL=re.compile(r'^\s*(EFX_\w+)\s+(\S+)\s*\((.*?)\) /\*[^\n]*',re.M|re.S)
OUTPUT={'EFX_LUT4':'O','EFX_FF':'Q'}
REF=re.compile(r"(?P<const>\d+'[sS]?[hHbBdD][0-9a-fA-FxXzZ_]+)|(?P<id>\\\S+|[A-Za-z_$][\w$]*)\s*(?:\[(?P<hi>\d+)(?::(?P<lo>\d+))?\])?")

def lexical_tokens(expr):
    result=[]
    for m in REF.finditer(expr):
        if m['const']:result.append(('literal',m['const'],None));continue
        name=m['id'].lstrip('\\')
        if m['hi'] is None:result.append(('net',name,None));continue
        hi=int(m['hi']);lo=hi if m['lo'] is None else int(m['lo']);step=-1 if hi>lo else 1
        result.extend(('net',name,i) for i in range(hi,lo+step,step))
    assert not REF.sub('',expr).strip(' ,{}\t\r\n'),('unsupported port expression',expr[:150])
    return result

def render_token(t):
    kind,name,bit=t
    if kind=='literal':return name
    # Always terminate the escaped identifier BEFORE any bit select.
    return '\\'+name+' '+('['+str(bit)+']' if bit is not None else '')

def rewrite_expression(expr,aliases):
    def replace(m):
        ts=lexical_tokens(m[0])
        if not any(t in aliases for t in ts):return m[0]
        values=[render_token(aliases.get(t,t)) for t in ts]
        return values[0] if len(values)==1 else '{'+', '.join(values)+'}'
    return REF.sub(replace,expr)

def raw_cells(text):
    return {norm(m[2]):(m[1],dict(re.findall(r'\.(\w+)\(([^()]*)\)',m[3])),m[3]) for m in CELL.finditer(text)}

def graph(text):
    return {n:(k,{p:lexical_tokens(v) for p,v in pins.items()}) for n,(k,pins,_) in raw_cells(text).items()}

def audit(original,changed,rows):
    a,b=graph(original),graph(changed);ap,bp=params(original),params(changed)
    aliases={('net',x['net'],None):a[r['driver']][1][OUTPUT[r['kind']]][0] for r in rows for x in r['clones']}
    clones={x['name']:r['driver'] for r in rows for x in r['clones']}
    assert len(aliases)==len(clones) and set(b)-set(a)==set(clones) and not set(a)-set(b)
    def collapse(pins):return {p:[aliases.get(t,t) for t in ts] for p,ts in pins.items()}
    for n,(kind,pins) in a.items():
        assert b[n][0]==kind and collapse(b[n][1])==pins,('graph',n)
        assert ap.get(n,{})==bp.get(n,{}),('params',n)
    for n,parent in clones.items():
        assert b[n][0]==a[parent][0] and collapse(b[n][1])==a[parent][1],('clone',n)
        assert bp[n]==ap[parent],('clone parameters',n)
        if b[n][0]=='EFX_FF':
            assert b[n][1]['CLK']==[('net','process_clk_150',None)] and bp[n]['SR_SYNC']==(1,1)
    assert protected_blocks(original)==protected_blocks(changed)
    for pattern in (r'^\s*(?:input|output|inout)\b.*$',r'^\s*assign\b.*$'):
        assert re.findall(pattern,original,re.M)==re.findall(pattern,changed,re.M)
    return dict(status='REVERSIBLE_LOCAL_DRIVER_GRAPH_PASS',original_cells=len(a),clones=len(clones),
        added_kinds=dict(collections.Counter(b[n][0] for n in clones)),
        identical_clock_reset_ce_and_init_parameters=True,latency_cycles_added=0,
        escaped_identifiers_and_bit_selects_distinguished=True,
        protected_blocks_byte_equal=True,hardware_access=False,physical_pass=False)

def partition(loads,positions):
    if len(loads)<2:return [loads]
    extents=[max(positions[n][d] for n,p in loads)-min(positions[n][d] for n,p in loads) for d in (0,1)]
    if len(loads)<=24 and max(extents)<=64:return [loads]
    axis=0 if extents[0]>=extents[1] else 1
    ordered=sorted(loads,key=lambda x:(positions[x[0]][axis],positions[x[0]][1-axis],x))
    mid=len(ordered)//2
    return partition(ordered[:mid],positions)+partition(ordered[mid:],positions)

def transform(source,placement,survey):
    c=cells(source);ps=params(source);raw=raw_cells(source)
    pos={}
    for line in placement.splitlines():
        parts=line.split()
        if len(parts)==4 and not line.startswith('#'):pos[parts[0]]=tuple(map(int,parts[1:3]))
    drivers={}
    for n,(kind,pins,attrs) in c.items():
        if kind in OUTPUT:drivers[pins[OUTPUT[kind]]]=(n,kind,pins,attrs)
    hot=collections.defaultdict(list)
    for row in survey['paths']:
        if row['slack_ns']>=0:continue
        for net in row['nets']:
            if net['fanout']>=12 and net['delay_ns']>=1.5:
                hot[net['net']].append(dict(slack_ns=row['slack_ns'],delay_ns=net['delay_ns'],fanout=net['fanout']))
    candidates={net:drivers[net] for net in hot if net in drivers and not drivers[net][0].startswith('protected_inst/')}
    loads=collections.defaultdict(list);unsafe=set()
    for n,(kind,pins,attrs) in c.items():
        for p,v in pins.items():
            for t in set(tokens(v)) & candidates.keys():
                if n==candidates[t][0] and p==OUTPUT[kind]:continue
                if (n.startswith('protected_inst/') or n not in pos or
                    re.search(r'CLK(?!E)|RESET|RST|^SR$',p) or 'async_reg' in attrs):unsafe.add(t)
                loads[t].append((n,p))
    edits=collections.defaultdict(dict);rows=[];new=[];declarations=[]
    for net,(driver,kind,pins,attrs) in sorted(candidates.items(),key=lambda kv:(min(x['slack_ns'] for x in hot[kv[0]]),kv[0])):
        if net in unsafe or driver not in pos:continue
        if kind=='EFX_FF' and (pins['CLK']!='process_clk_150' or ps[driver].get('SR_SYNC')!=(1,1) or 'async_reg' in attrs):continue
        groups=partition(loads[net],pos)
        if len(groups)<2:continue
        def distance(group):return sum(abs(pos[n][0]-pos[driver][0])+abs(pos[n][1]-pos[driver][1]) for n,p in group)/len(group)
        groups.sort(key=distance)
        row=dict(net=net,driver=driver,kind=kind,load_count=len(loads[net]),
            evidence=hot[net],original_loads=groups[0],clones=[])
        for group in groups[1:]:
            idx=len(new);name=f'LOCAL_DRIVER_{idx:03d}';wire=f'local_driver_net_{idx:03d}'
            assert name not in c and wire not in drivers
            npins=dict(raw[driver][1]);npins[OUTPUT[kind]]=wire
            new.append((name,kind,npins,ps[driver]))
            declarations.append(f'    (* syn_keep="true" *) wire {wire};')
            for n,p in group:edits[n].setdefault(p,{})[net]=wire
            row['clones'].append(dict(name=name,net=wire,loads=group,
                prior_bbox=[min(pos[n][d] for n,p in group) for d in (0,1)]+[max(pos[n][d] for n,p in group) for d in (0,1)]))
        rows.append(row)
    assert rows and 0<len(new)<=320,('replication budget',len(new))
    changed_count=0
    def replace(m):
        nonlocal changed_count
        name=norm(m[2])
        if name not in edits:return m[0]
        def port(q):
            p,v=q.groups()
            if p not in edits[name]:return q[0]
            mapping=edits[name][p]
            aliases={lexical_tokens(raw[drivers[t][0]][1][OUTPUT[drivers[t][1]]])[0]:('net',w,None) for t,w in mapping.items()}
            assert set(aliases)<=set(lexical_tokens(v)),('ambiguous selected net',name,p)
            return '.'+p+'('+rewrite_expression(v,aliases)+')'
        body=re.sub(r'\.(\w+)\(([^()]*)\)',port,m[3]);changed_count+=1
        return m[0][:m.start(3)-m.start()]+body+m[0][m.end(3)-m.start():]
    changed=CELL.sub(replace,source);assert changed_count==len(edits)
    for name,kind,pins,par in new:
        declarations.append('    '+kind+' '+name+' ('+', '.join('.'+p+'('+v+')' for p,v in pins.items())+') /* local identical driver */ ;')
        declarations.extend(f"    defparam {name}.{p} = {width}'h{value:x};" for p,(width,value) in par.items())
    assert len(re.findall(r'^endmodule\b',changed,re.M))==1
    changed=re.sub(r'^endmodule\b',lambda m:'\n'.join(declarations)+'\nendmodule',changed,flags=re.M)
    proof=audit(source,changed,rows)
    proof['selected_nets']=len(rows);proof['changed_load_cells']=changed_count
    proof['max_loads_per_new_driver']=max(len(x['loads']) for r in rows for x in r['clones'])
    proof['covered_negative_endpoints']=sum(any(n['net'] in {r['net'] for r in rows} for n in p['nets']) for p in survey['paths'] if p['slack_ns']<0)
    return changed,rows,proof

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args()
    root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
    base=ROOT/'tmp/frame-execution/native-split-timing-place1'
    source=base/'outflow/ti60f225_oob.map.v';place=base/'outflow/ti60f225_oob.out.placeloc'
    survey=ROOT/'tmp/frame-execution/native-split-timing-path-survey1/data_arc_analysis.json'
    try:
        original=source.read_text();changed,rows,proof=transform(original,place.read_text(),json.loads(survey.read_text()))
        (root/'native_source.v').write_text(changed,encoding='utf8')
        (root/'replications.json').write_text(json.dumps(rows,indent=2))
        # Negative checks must reject a changed LUT and an incorrect fanout alias.
        first=rows[0]['clones'][0]['name']
        bad=re.sub(r'(defparam '+first+r'\.LUTMASK\s*=\s*)[^;]+;',r"\g<1>16'hffff;",changed,count=1)
        assert bad!=changed
        for label,text in [('mask',bad),('wire',changed.replace('.O('+rows[0]['clones'][0]['net']+')','.O(1\'b0)',1))]:
            assert text!=changed,label
            try:audit(original,text,rows)
            except (AssertionError,KeyError):pass
            else:raise AssertionError('Corruption accepted: '+label)
        proof.update(negative_checks=['mask','wire'],inputs={str(p):sha(p) for p in (source,place,survey,Path(__file__))},
            source_sha256=sha(root/'native_source.v'),replications_sha256=sha(root/'replications.json'))
        (root/'result.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof),flush=True)
    except Exception as e:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise

if __name__=='__main__':main()
