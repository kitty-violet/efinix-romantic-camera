"""Keep the guarded RQ cut, localize its newly concentrated decode fanout."""
import argparse,copy,json,pickle,re,shutil,time
from pathlib import Path
from run_units import ROOT,sha
from native_local_drivers import CELL,render_token,partition
from camera_joint_native import OUT


def main():
    p=argparse.ArgumentParser()
    for n in ('base','placement','root'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();base=a.base.resolve();out=a.root.resolve();placement=a.placement.resolve()
    assert out.is_relative_to(ROOT/'tmp/frame-execution') and not out.exists()
    start=time.monotonic();proof=json.loads((base/'native-graph-audit.json').read_text())
    assert proof['status']=='SERVICE_NATIVE_RETAINED_GRAPH_AND_CUT_PASS'
    for f,h in proof['inputs'].items():assert sha(f)==h,f
    key,g,ps=pickle.loads((base/'rq-audit-parse.pkl').read_bytes())
    assert key[0]==sha(base/'native_source.v')
    target='LUT__140273';kind,pins=g[target];assert kind=='EFX_LUT4'
    token=pins['O'][0];loads=[(n,p) for n,(k,ports) in g.items() for p,ts in ports.items() if p not in OUT[k] and token in ts]
    assert len(loads)==240 and all(g[n][0]=='EFX_LUT4' and re.fullmatch('I[0-3]',p) for n,p in loads)
    pos={}
    for line in placement.read_text().splitlines():
        fields=line.split()
        if len(fields)==4 and not line.startswith('#'):
            pos[fields[0].lstrip('\\')]=tuple(map(int,fields[1:3]))
    assert all(n in pos for n,p in loads)
    groups=partition(loads,pos);assert 2<=len(groups)<=40
    source=(base/'native_source.v').read_text();edits={};clones=[];declarations=[];code='\n'
    for i,group in enumerate(groups):
        name=f'SERVICE_RQ_LOCAL_{i:02d}';net=('net',f'service_rq_local_{i:02d}',None)
        assert name not in g and net[1] not in source
        cp=copy.deepcopy(pins);cp['O']=[net]
        text='    EFX_LUT4 '+name+' ('+', '.join('.'+p+'('+render_token(ts[0])+')' for p,ts in cp.items())+') /* local RQ decode, no added cycle */ ;\n'
        text+='    defparam '+name+f".LUTMASK = 16'h{ps[target]['LUTMASK'][1]:04x};\n"
        decl='    (* syn_keep="true" *) wire '+net[1]+';\n';declarations.append(decl);code+=text
        clones.append(dict(name=name,token=net,loads=group,cell=text,declaration=decl,params=ps[target],pins=cp,
                           reference_bbox=[min(pos[n][d] for n,p in group) for d in (0,1)]+[max(pos[n][d] for n,p in group) for d in (0,1)]))
        for n,p in group:edits.setdefault(n,{})[p]=net
    records=[]
    def edit(m):
        n=m[2].lstrip('\\')
        if n not in edits:return m[0]
        cp=copy.deepcopy(g[n][1])
        for p,net in edits[n].items():assert cp[p]==[token];cp[p]=[net]
        ports=', '.join('.'+p+'('+(', '.join(render_token(t) for t in ts) if len(ts)<=1 else '{'+', '.join(render_token(t) for t in ts)+'}')+')' for p,ts in cp.items())
        after=m[0][:m.start(3)-m.start()]+ports+m[0][m.end(3)-m.start():]
        records.append(dict(name=n,before=m[0],after=after));return after
    text=CELL.sub(edit,source);assert len(records)==len(edits)
    index=CELL.search(text).start();text=text[:index]+''.join(declarations)+text[index:]
    index=text.index('    protected_cell protected_inst (');text=text[:index]+code+text[index:]
    out.mkdir()
    for f in base.glob('*.sv'):shutil.copy2(f,out/f.name)
    (out/'native_source.v').write_text(text,encoding='utf8')
    receipt=dict(base=str(base),base_sha256=sha(base/'native_source.v'),parent_proof_sha256=sha(base/'native-graph-audit.json'),
                 target=target,target_token=token,placement=str(placement),placement_sha256=sha(placement),clones=clones,records=records,
                 inserted_code=code,inserted_declarations=''.join(declarations),builder_sha256=sha(__file__),
                 reason='actual mapped decoder fanout 193->433 after RQ cut; preserve removed correction, separate decode loads geometrically',
                 parent_pack_xlr=60103,physical_improvement_proven=False,hardware_access=False)
    (out/'rq-local-edit.json').write_text(json.dumps(receipt,indent=2))
    graft=json.loads((base/'graft.json').read_text());graft['rq_local']=dict(base=str(base),manifest=str(out/'rq-local-edit.json'),manifest_sha256=sha(out/'rq-local-edit.json'))
    graft['inputs'][str(Path(__file__).resolve())]=sha(__file__)
    (out/'graft.json').write_text(json.dumps(graft,indent=2))
    print(json.dumps(dict(status='RQ_LOCAL_CANDIDATE_PENDING_PROOF',clones=len(clones),loads=len(loads),max_group=max(map(len,groups)),seconds=time.monotonic()-start)))

if __name__=='__main__':main()
