"""Independent guarded RQ specialization; fixed B1 graph and A stay unchanged.

Nonzero MAC scales representable in signed7 are rejected, not approximated.
Feedback scales remain fully signed18. This removes the unneeded correction
pass while preserving signed32 overflow, saturation, ReLU and queue ownership.
No claim of physical release is made by this generator.
"""
import argparse, collections, copy, json, re, shutil, time
from pathlib import Path
from run_units import ROOT, HERE, sha
from video_service_graph import load
from native_local_drivers import CELL, REF, render_token, rewrite_expression
from camera_joint_native import OUT
from complete_native_logic_split import protected_blocks

P='video_processing_pipeline/pipeline/cnn/core/dut/engine/rq/'
ZERO=('literal',"1'b0",None)
ONE=('literal',"1'b1",None)

def transform(base):
    d=load();graft=json.loads((base/'graft.json').read_text());edits={}
    for e in graft['changed_loads']:
        edits.setdefault((e['cell'],e['pin']),{})[tuple(e['from_token'])]=tuple(e['to_token'])
    old={n:(k,{p:[edits.get((n,p),{}).get(t,t) for t in ts] for p,ts in pins.items()})
         for n,(k,pins) in d['g'].items() if n not in graft['removed']}
    g=copy.deepcopy(old);ps=copy.deepcopy(d['ps']);aliases={};removed=set();changed=set();guards={}
    forced={P+'preparing~FF',P+'correction_wait[0]~FF',P+'correction_wait[1]~FF'}
    forced.update(P+f'selected_split[{i}]~FF' for i in range(32))
    for n in forced:
        k,pins=g[n];assert k=='EFX_FF' and len(pins['Q'])==1
        assert ps[n]['SR_VALUE']==(1,0) and ps[n].get('INIT',(1,0))==(1,0)
        aliases[pins['Q'][0]]=ZERO;removed.add(n)
    # Every ordinary scale error consumer receives the new error. The class
    # registers themselves are kept, including their preparation/clock edge.
    scale_rewrites={}
    for c in range(8):
        ns=[P+f'{field}[{c}]~FF' for field in ('scale_invalid_q','scale_fits7_q','scale_nonzero_q')]
        ts=[g[n][1]['Q'][0] for n in ns]
        out=('net',f'service_rq_scale_invalid_{c}',None)
        guards[f'SERVICE_RQ_SCALE_GUARD_{c}']=('EFX_LUT4',
            {**{f'I{i}':[t] for i,t in enumerate(ts)},'I3':[ZERO],'O':[out]})
        scale_rewrites[ts[0]]=out
    for n,(k,pins) in g.items():
        if n in forced:continue
        for p,ts in pins.items():
            if p not in OUT[k] and any(t in scale_rewrites for t in ts):
                pins[p]=[scale_rewrites.get(t,t) for t in ts];changed.add(n)
    def resolved(t):
        visited=set()
        while t in aliases:
            assert t not in visited;visited.add(t);t=aliases[t]
        return t
    cofactors={};merges=[]
    while True:
        before=(len(removed),len(aliases),len(cofactors))
        for n,(k,pins) in g.items():
            if n in removed:continue
            touch=False
            for p,ts in pins.items():
                if p not in OUT[k]:
                    new=[resolved(t) for t in ts]
                    if new!=ts:pins[p]=new;changed.add(n);touch=True
            if k!='EFX_LUT4' or not (touch or n in cofactors):continue
            mask=ps[n]['LUTMASK'][1];ins=[pins.get(f'I{i}',[ZERO])[0] for i in range(4)]
            assert all(t[0]!='literal' or t in (ZERO,ONE) for t in ins)
            # Only constant substitution; repeated/X/Z nets are NOT Boolean
            # don't-cares. Pin positions and unknown propagation are retained.
            newmask=0
            for x in range(16):
                y=sum((int(t==ONE) if t[0]=='literal' else (x>>i)&1)<<i for i,t in enumerate(ins))
                newmask|=((mask>>y)&1)<<x
            if newmask!=mask:
                cofactors.setdefault(n,dict(before_mask=d['ps'][n]['LUTMASK'][1],before=old[n][1]))
                ps[n]['LUTMASK']=(16,newmask);changed.add(n)
            if newmask in (0,65535):
                aliases[pins['O'][0]]=ONE if newmask else ZERO;removed.add(n)
        # Merge identical affected LUTs only, retaining a real LUT for X/Z
        # behavior. In particular a LUT buffer is never replaced by a wire.
        groups=collections.defaultdict(list)
        for n in changed-removed:
            k,pins=g[n]
            if k=='EFX_LUT4':
                mask=ps[n]['LUTMASK'][1]
                support=[i for i in range(4) if any(((mask>>v)^(mask>>(v^(1<<i))))&1 for v in range(16))]
                if len(support)>1:continue
                key=(mask,tuple(tuple(pins.get(f'I{i}',[ZERO])) if i in support else (ZERO,) for i in range(4)))
                groups[key].append(n)
        for ns in groups.values():
            if len(ns)<2:continue
            ns.sort();keep=ns[0]
            for n in ns[1:]:
                aliases[g[n][1]['O'][0]]=g[keep][1]['O'][0];removed.add(n)
                merges.append(dict(removed=n,kept=keep))
        if before==(len(removed),len(aliases),len(cofactors)):break
    aliases={t:resolved(v) for t,v in aliases.items()}
    for n,(k,pins) in g.items():
        if n in removed:continue
        for p,ts in pins.items():
            if p not in OUT[k]:pins[p]=[aliases.get(t,t) for t in ts]
    source=(base/'native_source.v').read_text();records=[]
    # Remove only unobserved ancestors of this correction cut. Other old
    # dead cones are deliberately outside this candidate's scope.
    drivers={t:n for n,(k,pins) in old.items() for p,ts in pins.items() if p in OUT[k] for t in ts}
    ancestors=set(removed);todo=list(removed)
    while todo:
        n=todo.pop();k,pins=old[n]
        for p,ts in pins.items():
            if p in OUT[k] or p in ('CLK','SR'):continue
            for t in ts:
                parent=drivers.get(t)
                if parent and parent not in ancestors and old[parent][0] in ('EFX_LUT4','EFX_ADD'):
                    ancestors.add(parent);todo.append(parent)
    outer=CELL.sub('',source)
    outer=re.sub(r'^\s*defparam\b[^;]*;','',outer,flags=re.M)
    outer=re.sub(r'^\s*wire\b[^;]*;','',outer,flags=re.M)
    roots={m['id'].lstrip('\\') for m in REF.finditer(outer) if m['id']}
    roots.update(t[1] for t in aliases.values() if t[0]=='net')
    roots.update(t[1] for k,pins in guards.values() for p,ts in pins.items() if p!='O' for t in ts if t[0]=='net')
    sdc=(ROOT/'tmp/frame-execution/video-service-place18-narrow/ti60f225_oob.sdc')
    if not sdc.exists():
        sdc=ROOT/'tmp/frame-execution/video-service-place20-narrow-c1/ti60f225_oob.sdc'
    assert sdc.exists();constraints=sdc.read_text()
    for n in ancestors:
        if n in constraints:roots.update(t[1] for p,ts in old[n][1].items() if p in OUT[old[n][0]] for t in ts if t[0]=='net')
    dead=[]
    while True:
        loads=collections.Counter(t for n,(k,pins) in g.items() if n not in removed for p,ts in pins.items() if p not in OUT[k] for t in ts)
        drop={n for n in ancestors-removed if g[n][0] in ('EFX_LUT4','EFX_ADD') and
              all(t[0]!='net' or (not loads[t] and t[1] not in roots) for p,ts in g[n][1].items() if p in OUT[g[n][0]] for t in ts)}
        if not drop:break
        removed.update(drop);dead.extend(sorted(drop))
    for n,(k,pins) in g.items():
        if n in removed:continue
        for p,ts in pins.items():
            if p in OUT[k]:continue
            for t in ts:
                assert drivers.get(t) not in removed or t in aliases,('removed live driver',n,p,t,drivers.get(t))
    for t in aliases.values():
        assert t[0]!='net' or drivers.get(t) not in removed,('removed alias driver',t)
    def replace(m):
        n=m[2].lstrip('\\')
        if n in removed:new=''
        elif n in changed:
            pins=g[n][1]
            new=m[0][:m.start(3)-m.start()]+', '.join('.'+p+'('+(', '.join(render_token(t) for t in ts) if len(ts)<=1 else '{'+', '.join(render_token(t) for t in ts)+'}')+')' for p,ts in pins.items())+m[0][m.end(3)-m.start():]
        else:return m[0]
        records.append(dict(name=n,before=m[0],after=new));return new
    text=CELL.sub(replace,source)
    def parameter(m):
        n=m[1].lstrip('\\')
        if n in removed:return ''
        if n in cofactors and m[2]=='LUTMASK':return f"\n    defparam {m[1]}.LUTMASK = 16'h{ps[n]['LUTMASK'][1]:04x};"
        return m[0]
    text=re.sub(r'^\s*defparam\s+(\S+?)\s*\.([A-Z_0-9]+)\s*=[^;]*;',parameter,text,flags=re.M)
    declarations='\n'.join('    wire '+render_token(p['O'][0])+';' for k,p in guards.values())
    anchor=CELL.search(text).start();text=text[:anchor]+'\n'+declarations+'\n'+text[anchor:]
    code='\n'
    for n,(k,pins) in guards.items():
        code+='    EFX_LUT4 '+n+' ('+', '.join('.'+p+'('+render_token(ts[0])+')' for p,ts in pins.items())+') /* guarded fast RQ */ ;\n'
        code+=f"    defparam {n}.LUTMASK = 16'heaea;\n"
    # Preserve all original named exports/declarations even when their driver
    # becomes a proved constant/alias. New assignments are unique drivers.
    code+='\n'.join('    assign '+render_token(t)+' = '+render_token(v)+';' for t,v in aliases.items())+'\n'
    anchor=text.index('    protected_cell protected_inst (');text=text[:anchor]+code+text[anchor:]
    assert protected_blocks(text)==protected_blocks(source)
    manifest=dict(base=str(base),base_sha256=sha(base/'native_source.v'),policy='MAC scale zero or outside signed7; feedback full signed18',
        forced_zero_state=sorted(forced),removed=sorted(removed),cofactors={n:{**r,'after_mask':ps[n]['LUTMASK'][1],'after':g[n][1]} for n,r in cofactors.items()},
        merges=merges,dead_correction_ancestors=dead,constraints=str(sdc),constraints_sha256=sha(sdc),aliases=[dict(source=t,target=v) for t,v in aliases.items()],guards=guards,
        changed_cells=sorted(changed-removed),records=records,numeric_proof_pending=True,physical_release=False)
    return text,manifest

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    base=a.base.resolve();out=a.root.resolve();assert out.is_relative_to(ROOT/'tmp') and not out.exists()
    start=time.monotonic();text,receipt=transform(base);out.mkdir()
    for f in base.glob('*.sv'):shutil.copy2(f,out/f.name)
    (out/'native_source.v').write_text(text,encoding='utf8')
    graft=json.loads((base/'graft.json').read_text());graft['rq_fast']=dict(base=str(base),manifest=str(out/'rq-fast-edit.json'),manifest_sha256=None)
    # Existing graft is history; the layered independent audit owns this new
    # modification and never falsely marks it an unchanged retained graph.
    (out/'rq-fast-edit.json').write_text(json.dumps(receipt,indent=2))
    graft['rq_fast']['manifest_sha256']=sha(out/'rq-fast-edit.json')
    graft['inputs'][str(Path(__file__))]=sha(__file__)
    (out/'graft.json').write_text(json.dumps(graft,indent=2))
    print(json.dumps(dict(status='RQ_FAST_CANDIDATE_PENDING_PROOF',removed=len(receipt['removed']),forced=len(receipt['forced_zero_state']),
        cofactors=len(receipt['cofactors']),merges=len(receipt['merges']),guards=8,seconds=time.monotonic()-start,hardware_access=False)))

if __name__=='__main__':main()
