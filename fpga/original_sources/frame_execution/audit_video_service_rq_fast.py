"""Independent cut, local four-state and accepted-A invariant audit."""
import argparse,collections,itertools,json,re,sys,time,pickle,functools
from pathlib import Path
from run_units import ROOT,HERE,sha
from video_service_graph import load
from audit_continuous_vectors import expand_graph
from probe_full_native_split_map import params
from complete_native_logic_split import protected_blocks
from camera_joint_native import OUT
from native_local_drivers import CELL,REF,render_token

P='video_processing_pipeline/pipeline/cnn/core/dut/engine/rq/'
Z=('literal',"1'b0",None);O=('literal',"1'b1",None)

def lut(mask,ins,values):
    # Official EFX_LUT4 optional-input weak0 turns Z into 0; X explores
    # both branches independently, including repeated occurrences.
    possibilities=[(0,) if t==Z else (1,) if t==O else (0,1) if values.get(t,'x')=='x' else (0,) if values.get(t)=='z' else (values[t],) for t in ins]
    outcomes={mask>>sum(b<<i for i,b in enumerate(bits))&1 for bits in itertools.product(*possibilities)}
    return next(iter(outcomes)) if len(outcomes)==1 else 'x'

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--numeric',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();start=time.monotonic();graft=json.loads((root/'graft.json').read_text());meta=graft['rq_fast'];receipt=Path(meta['manifest'])
    assert sha(receipt)==meta['manifest_sha256'];r=json.loads(receipt.read_text());base=Path(r['base'])
    assert sha(base/'native_source.v')==r['base_sha256']
    previous=json.loads((base/'native-graph-audit.json').read_text());assert previous['status']=='SERVICE_NATIVE_RETAINED_GRAPH_AND_CUT_PASS'
    for f,h in previous['inputs'].items():assert sha(f)==h
    numeric=a.numeric.resolve();nr=json.loads(numeric.read_text());assert nr['status']=='GUARDED_FAST_RQ_INT32_NUMERIC_QUEUE_CANCEL_PASS'
    for f,h in nr['inputs'].items():assert sha(f)==h
    sys.path[:0]=[str(ROOT),str(ROOT/'network_ab'),str(ROOT/'network_ab/training')]
    from b1_trained_image import decode
    image=ROOT/'model-training-workspace/accepted/B1-A-20261008/coefficients-padded.bin'
    assert sha(image)=='ceebe7b3187fd47fbf0924883d19a9a335923d9de378ae09d30f362d9b9d0523'
    coefficients,residual=decode(image.read_bytes())
    scales=[v for op in coefficients.values() for v in op.multiplier]
    assert min(scales)==421 and max(scales)==131071 and len(scales)==563
    assert all(not(-64<=v<=63) for v in scales)
    assert all(v==dict(main_scaled=False,multiplier=1,post_multiplier=131071) for v in residual.values())
    d=load();og=json.loads((base/'graft.json').read_text());edits={}
    for e in og['changed_loads']:edits.setdefault((e['cell'],e['pin']),{})[tuple(e['from_token'])]=tuple(e['to_token'])
    old={n:(k,{p:[edits.get((n,p),{}).get(t,t) for t in ts] for p,ts in pins.items()}) for n,(k,pins) in d['g'].items() if n not in og['removed']}
    text=(root/'native_source.v').read_text();cache=root/'rq-audit-parse.pkl';key=[sha(root/'native_source.v'),sha(HERE/'audit_continuous_vectors.py'),sha(HERE/'probe_full_native_split_map.py')]
    if cache.exists():
        cached,new,np=pickle.loads(cache.read_bytes());assert cached==key
    else:
        new=expand_graph(text);np=params(text);cache.write_bytes(pickle.dumps((key,new,np),protocol=4))
    removed=set(r['removed']);forced=set(r['forced_zero_state']);dead=set(r['dead_correction_ancestors'])
    assert len(forced)==35 and set(new)==set(old)-removed|set(r['guards'])
    aliases={tuple(e['source']):tuple(e['target']) for e in r['aliases']}
    assert len(aliases)==len(removed-dead)
    replacement={old[P+f'scale_invalid_q[{c}]~FF'][1]['Q'][0]:('net',f'service_rq_scale_invalid_{c}',None) for c in range(8)}
    def sub(t):return aliases.get(replacement.get(t,t),replacement.get(t,t))
    forced_q={old[n][1]['Q'][0] for n in forced}
    # Inductive proof in the ORIGINAL native graph: after prepared A headers,
    # every removed state bit stays zero, for arbitrary other Boolean inputs.
    values={t:0 for t in forced_q}
    values.update({old[P+f'scale_fits7_q[{c}]~FF'][1]['Q'][0]:0 for c in range(8)})
    # First propagate only definite constants. Do not expand unrelated
    # arithmetic/control merely to simplify AND-with-zero.
    comb=[(n,k,pins) for n,(k,pins) in old.items() if k in ('EFX_LUT4','EFX_ADD')]
    for iteration in range(100):
        count=len(values)
        for n,k,pins in comb:
            if k=='EFX_LUT4':
                out=pins['O'][0]
                if out in values:continue
                ins=[pins.get('I'+str(i),[Z])[0] for i in range(4)]
                v=lut(d['ps'][n]['LUTMASK'][1],ins,values)
                if v!='x':values[out]=v
            else:
                par=d['ps'][n];ins=[pins.get(i,[Z])[0] for i in ('I0','I1','CI')]
                v=[0 if t==Z else 1 if t==O else values.get(t) for t in ins]
                if None in v:continue
                total=(v[0]^(1-par.get('I0_POLARITY',(1,1))[1]))+(v[1]^(1-par.get('I1_POLARITY',(1,1))[1]))+v[2]
                for pin,val in [('O',total&1),('CO',total>>1)]:
                    for out in pins.get(pin,[]):values.setdefault(out,val)
        if len(values)==count:break
    else:raise AssertionError('constant propagation did not converge')
    # Reduced ordered decision diagrams preserve correlations between D,
    # CE and reset. Checking their constants independently is insufficient
    # for a clock-enabled FF (e.g. CE=x, D=!x still always loads zero).
    drivers={t:(n,p) for n,(k,pins) in old.items() for p,ts in pins.items() if p in OUT[k] for t in ts}
    # A bounded local cut is an over-approximation: all exterior signals are
    # independent free variables, never assumptions about their values.
    roots=[t for n in forced for p in ('D','CE') for t in old[n][1].get(p,[])]
    local=set();front=set(roots)
    for depth in range(1):
        next_front=set()
        for t in front:
            if t in values or t in (Z,O):continue
            info=drivers.get(t)
            if info and old[info[0]][0] in ('EFX_LUT4','EFX_ADD'):
                n,p=info;local.add(n);k,pins=old[n]
                next_front.update(x for pin,ts in pins.items() if pin not in OUT[k] for x in ts)
        front=next_front
    nodes={};unique={};variables={}
    def node(v,lo,hi):
        if lo==hi:return lo
        key=(v,lo,hi)
        if key not in unique:
            assert len(nodes)<100000,'bounded local induction proof node limit'
            i=len(nodes)+2;unique[key]=i;nodes[i]=key
        return unique[key]
    @functools.lru_cache(None)
    def ite(c,t,f):
        if c<2:return t if c else f
        if t==f:return t
        if t==1 and f==0:return c
        v=min(nodes[x][0] for x in (c,t,f) if x>=2)
        def part(x,high):return nodes[x][2 if high else 1] if x>=2 and nodes[x][0]==v else x
        return node(v,ite(part(c,False),part(t,False),part(f,False)),ite(part(c,True),part(t,True),part(f,True)))
    def inv(x):return ite(x,0,1)
    @functools.lru_cache(None)
    def expression(t):
        if t in (Z,O):return int(t==O)
        if t in values:return values[t]
        info=drivers.get(t)
        if info and info[0] in local:
            n,p=info;k,pins=old[n];par=d['ps'][n]
            if k=='EFX_LUT4':
                ins=[expression(pins.get('I'+str(i),[Z])[0]) for i in range(4)]
                def table(depth,offset):
                    if depth<0:return (par['LUTMASK'][1]>>offset)&1
                    return ite(ins[depth],table(depth-1,offset|(1<<depth)),table(depth-1,offset))
                return table(3,0)
            aa,bb,cc=[expression(pins.get(i,[Z])[0]) for i in ('I0','I1','CI')]
            if not par.get('I0_POLARITY',(1,1))[1]:aa=inv(aa)
            if not par.get('I1_POLARITY',(1,1))[1]:bb=inv(bb)
            xor=ite(aa,inv(bb),bb)
            return ite(xor,inv(cc),cc) if p=='O' else ite(aa,ite(bb,1,cc),ite(bb,cc,0))
        if t not in variables:variables[t]=len(variables)
        return node(variables[t],0,1)
    invariants=[]
    for n in forced:
        k,pins=old[n];par=d['ps'][n]
        assert k=='EFX_FF' and par['SR_VALUE']==(1,0) and par.get('INIT',(1,0))==(1,0)
        dv=expression(pins['D'][0]);cv=expression(pins.get('CE',[O])[0])
        if not par.get('D_POLARITY',(1,1))[1]:dv=inv(dv)
        if not par.get('CE_POLARITY',(1,1))[1]:cv=inv(cv)
        # Every reset writes zero, so proving the non-reset transition is
        # sufficient regardless of reset polarity or synchronous priority.
        nxt=ite(cv,dv,0);assert nxt==0,('zero-state invariant',n,nxt,len(nodes))
        assert aliases[pins['Q'][0]]==Z
        invariants.append(dict(cell=n,next_state_bdd=nxt,reset_value=0,initial_value=0))
    merged={e['removed']:e['kept'] for e in r['merges']};checks=0;exact=0
    for n,(kind,pins) in old.items():
        if n in forced or n in dead:continue
        expected={p:[t if p in OUT[kind] else sub(t) for t in ts] for p,ts in pins.items()}
        if n in removed:
            assert kind=='EFX_LUT4'
            constant=aliases[pins['O'][0]]
            if constant not in (Z,O):
                assert n in merged
                out=aliases[pins['O'][0]];drivers=[x for x,(k,p) in new.items() if k=='EFX_LUT4' and p['O']==[out]]
                assert len(drivers)==1,(n,out,drivers,merged.get(n));target=drivers[0];actual=new[target][1];mask=np[target]['LUTMASK'][1]
            else:actual=None;mask=None;constant=aliases[pins['O'][0]];assert constant in (Z,O)
        else:
            assert new[n][0]==kind
            if kind!='EFX_LUT4':
                assert new[n][1]==expected and np.get(n)==d['ps'].get(n),n;exact+=1;continue
            actual=new[n][1];mask=np[n]['LUTMASK'][1];assert actual['O']==pins['O']
        ins=[expected.get('I'+str(i),[Z])[0] for i in range(4)]
        outs=[actual.get('I'+str(i),[Z])[0] for i in range(4)] if actual else []
        variables=list(dict.fromkeys(t for t in ins+outs if t[0]=='net'));assert len(variables)<=4,(n,variables)
        if n not in removed and expected==actual and mask==d['ps'][n]['LUTMASK'][1]:exact+=1;continue
        for vector in itertools.product((0,1,'x','z'),repeat=len(variables)):
            vals=dict(zip(variables,vector));left=lut(d['ps'][n]['LUTMASK'][1],ins,vals)
            right=lut(mask,outs,vals) if actual else int(constant==O)
            assert left==right,('four-state cofactor',n,vector,left,right);checks+=1
    # The new guards use only the actual three existing class FF outputs.
    for c in range(8):
        n=f'SERVICE_RQ_SCALE_GUARD_{c}';assert new[n][0]=='EFX_LUT4' and np[n]=={'LUTMASK':(16,0xeaea)}
        expected={f'I{i}':[old[P+f'{field}[{c}]~FF'][1]['Q'][0]] for i,field in enumerate(('scale_invalid_q','scale_fits7_q','scale_nonzero_q'))}
        expected.update(I3=[Z],O=[('net',f'service_rq_scale_invalid_{c}',None)])
        assert new[n][1]==expected
    # No pruned ancestor can feed retained logic, a guard, the encrypted IP,
    # a behavioral module, an alias assignment or a timing constraint.
    outer=CELL.sub('',text);outer=re.sub(r'^\s*defparam\b[^;]*;','',outer,flags=re.M);outer=re.sub(r'^\s*wire\b[^;]*;','',outer,flags=re.M)
    refs={m['id'].lstrip('\\') for m in REF.finditer(outer) if m['id']}
    loads={t for k,pins in new.values() for p,ts in pins.items() if p not in OUT[k] for t in ts}
    for n in dead:
        assert old[n][0] in ('EFX_LUT4','EFX_ADD')
        for p,ts in old[n][1].items():
            if p in OUT[old[n][0]]:
                assert all(t not in loads and (t[0]!='net' or t[1] not in refs) for t in ts),n
    assert protected_blocks(text)==protected_blocks((base/'native_source.v').read_text())
    for f in base.glob('*.sv'):assert sha(f)==sha(root/f.name)
    assert sha(r['constraints'])==r['constraints_sha256']
    # Exterior changes are exactly declared guard nets and stated aliases;
    # original module bodies/assignments/topology are byte-preserved.
    def exterior(s):
        s=CELL.sub('',s);s=re.sub(r'^\s*defparam\b[^;]*;','',s,flags=re.M);return re.sub(r'\s+','',s)
    stripped=text
    for c in range(8):stripped=stripped.replace('    wire '+render_token(('net',f'service_rq_scale_invalid_{c}',None))+';','')
    for x,y in aliases.items():stripped=stripped.replace('    assign '+render_token(x)+' = '+render_token(y)+';','')
    assert exterior(stripped)==exterior((base/'native_source.v').read_text())
    result=dict(status='SERVICE_NATIVE_RETAINED_GRAPH_AND_CUT_PASS',scope='Guarded RQ policy specialization; NOT unrestricted old low-scale equivalence',
        retained_primitives=len(new),removed_primitives=len(removed),exact_unchanged_or_substituted=exact,
        four_state_local_comparisons=checks,original_native_zero_state_invariants=invariants,A_valid_MAC_channels=len(scales),A_scale_range=[min(scales),max(scales)],
        residual_parameters_unchanged=True,unsupported_nonzero_signed7_MAC_scale='reported by existing out_bad and downstream abort',
        numeric_short=str(numeric),seconds=time.monotonic()-start,
        inputs={str(f):sha(f) for f in (Path(__file__),receipt,root/'graft.json',root/'native_source.v',base/'native-graph-audit.json',numeric,image,ROOT/'network_ab/training/b1_trained_image.py')},
        hardware_access=False,board_release=False)
    (root/'native-graph-audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('inputs','original_native_zero_state_invariants')},indent=2))

if __name__=='__main__':main()
