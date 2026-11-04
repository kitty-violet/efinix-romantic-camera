"""Separate the measured long shared fault-reduction branch, without latency."""
import argparse,copy,itertools,json,re,shutil
from pathlib import Path
from run_units import ROOT,sha
from native_local_drivers import CELL,graph,render_token
from probe_full_native_split_map import params
from audit_video_service_rq_fast import lut

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--survey',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    base=a.base.resolve();out=a.root.resolve();survey=a.survey.resolve()
    assert out.is_relative_to(ROOT/'tmp') and not out.exists()
    previous=base/'native-graph-audit.json';parent=json.loads(previous.read_text())
    assert parent['status']=='SERVICE_NATIVE_RETAINED_GRAPH_AND_CUT_PASS'
    for f,h in parent['inputs'].items():assert sha(f)==h,f
    sr=json.loads(survey.read_text());assert sr['status']=='ROUTED_PATH_SURVEY_NOT_RELEASE'
    paths=sr['reports']['survey_setup.rpt']['paths']
    assert len(paths)==256 and all('engine/mac/werror~FF|CLK' in x['start'] and x['slack_ns']<0 for x in paths)
    text=(base/'native_source.v').read_text();names=['LUT__132940','LUT__132941','LUT__132942'];sink='LUT__132950'
    fragments={m[2].lstrip('\\'):m[0] for m in CELL.finditer(text) if m[2].lstrip('\\') in names+[sink]}
    assert set(fragments)==set(names+[sink])
    parlines={n:re.search(r'^\s*defparam '+n+r'\.LUTMASK\s*=\s*[^;]*;',text,re.M)[0] for n in names+[sink]}
    local='\n'.join(fragments[n]+parlines[n] for n in names+[sink]);g=graph(local);ps=params(local)
    aliases={};clones=[];decl='';insert='\n'
    for i,n in enumerate(names):
        assert g[n][0]=='EFX_LUT4'
        nm=f'SERVICE_FAULT_LOCAL_{i}';net=('net',f'service_fault_local_{i}',None)
        assert nm not in text and net[1] not in text
        pins=copy.deepcopy(g[n][1]);original=pins['O'][0]
        pins={p:([net] if p=='O' else [aliases.get(t,t) for t in ts]) for p,ts in pins.items()}
        aliases[original]=net
        body='    EFX_LUT4 '+nm+' ('+', '.join('.'+p+'('+render_token(ts[0])+')' for p,ts in pins.items())+') /* exact local fault reduction */ ;\n'
        body+=f"    defparam {nm}.LUTMASK = 16'h{ps[n]['LUTMASK'][1]:04x};\n"
        declaration='    (* syn_keep="true" *) wire '+net[1]+';\n';decl+=declaration;insert+=body
        clones.append(dict(parent=n,name=nm,cell=body,declaration=declaration,old_output=original,new_output=net))
    old=fragments[sink];target=g[names[-1]][1]['O'][0]
    assert g[sink][1]['I0']==[target]
    match=re.search(r'\.I0\([^()]*\)',old);assert match
    new=old[:match.start()]+'.I0('+render_token(aliases[target])+')'+old[match.end():]
    assert text.count(old)==1;changed=text.replace(old,new,1)
    pos=CELL.search(changed).start();changed=changed[:pos]+decl+changed[pos:]
    pos=changed.index('    protected_cell protected_inst (');changed=changed[:pos]+insert+changed[pos:]
    # Reconstruct the parent byte-for-byte; no hidden changes to state or RTL.
    restored=changed.replace(decl,'',1).replace(insert,'',1).replace(new,old,1);assert restored==text
    proofcells=graph(insert);proofparams=params(insert);inverse={v:k for k,v in aliases.items()};comparisons=0
    Z=('literal',"1'b0",None)
    for row in clones:
        n,nm=row['parent'],row['name'];assert proofcells[nm][0]=='EFX_LUT4' and proofparams[nm]==ps[n]
        pins={p:[inverse.get(t,t) for t in ts] for p,ts in proofcells[nm][1].items()};assert pins==g[n][1]
        ins=[pins.get('I'+str(i),[Z])[0] for i in range(4)]
        leaves=list(dict.fromkeys(t for t in ins if t[0]=='net'))
        for bits in itertools.product((0,1,'x','z'),repeat=len(leaves)):
            vals=dict(zip(leaves,bits));assert lut(proofparams[nm]['LUTMASK'][1],ins,vals)==lut(ps[n]['LUTMASK'][1],ins,vals);comparisons+=1
    assert graph(new)[sink][1]['I0']==[aliases[target]]
    out.mkdir()
    for f in base.glob('*.sv'):shutil.copy2(f,out/f.name);assert sha(f)==sha(out/f.name)
    (out/'native_source.v').write_text(changed,encoding='utf8')
    receipt=dict(base=str(base),base_sha256=sha(base/'native_source.v'),parent_proof_sha256=sha(previous),survey=str(survey),survey_sha256=sha(survey),clones=clones,sink=sink,before=old,after=new,insert=insert,declarations=decl,added_luts=3,added_state=0,latency_added=0,ordinary_fixed_locations_added=0)
    rp=out/'ready-local-edit.json';rp.write_text(json.dumps(receipt,indent=2))
    graft=json.loads((base/'graft.json').read_text());graft['ready_local']=dict(base=str(base),manifest=str(rp),manifest_sha256=sha(rp));graft['inputs'][str(Path(__file__).resolve())]=sha(__file__)
    (out/'graft.json').write_text(json.dumps(graft,indent=2))
    inputs={**parent['inputs'],**{str(f):sha(f) for f in (previous,Path(__file__).resolve(),survey,rp,out/'graft.json',out/'native_source.v')}}
    proof={**parent,'retained_primitives':parent['retained_primitives']+3,'ready_local_parent':str(base),'ready_local_clones':3,'ready_local_four_state_comparisons':comparisons,'ready_local_reversible_text':True,'inputs':inputs,'hardware_access':False,'board_release':False}
    (out/'native-graph-audit.json').write_text(json.dumps(proof,indent=2))
    print(json.dumps(dict(status=proof['status'],retained=proof['retained_primitives'],clones=3,comparisons=comparisons,latency_added=0,source_sha256=sha(out/'native_source.v'))))
if __name__=='__main__':main()
