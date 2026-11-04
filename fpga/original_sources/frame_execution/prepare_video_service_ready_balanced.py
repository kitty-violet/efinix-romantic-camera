"""Balance the proven AND cut so late MAC/RQ fault branches cross one gate."""
import argparse,itertools,json,re,shutil
from pathlib import Path
from run_units import ROOT,sha
from native_local_drivers import CELL,graph,render_token
from probe_full_native_split_map import params
from audit_video_service_rq_fast import lut

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 base=a.base.resolve();out=a.root.resolve();assert out.is_relative_to(ROOT/'tmp') and not out.exists()
 pp=base/'native-graph-audit.json';parent=json.loads(pp.read_text());assert parent['ready_local_clones']==3
 for f,h in parent['inputs'].items():assert sha(f)==h
 text=(base/'native_source.v').read_text();names=['SERVICE_FAULT_LOCAL_2','LUT__132950']
 blocks={m[2].lstrip('\\'):m[0] for m in CELL.finditer(text) if m[2].lstrip('\\') in names}
 pars={n:re.search(r'^\s*defparam '+n+r'\.LUTMASK\s*=[^;]*;',text,re.M)[0] for n in names}
 g=graph('\n'.join(blocks.values()));ps=params('\n'.join(pars.values()))
 assert all(g[n][0]=='EFX_LUT4' and ps[n]['LUTMASK']==(16,0x8000) for n in names)
 ap,bp=[g[n][1] for n in names];assert bp['I0']==ap['O']
 # Original: (a & b & MAC_OK & RQ_OK) & state1 & state2 & state3.
 # Balanced: (a & b & state1 & state2) & MAC_OK & RQ_OK & state3.
 newa={**ap,'I2':bp['I1'],'I3':bp['I2']}
 newb={**bp,'I1':ap['I2'],'I2':ap['I3']}
 leaves=list(dict.fromkeys(ap['I0']+ap['I1']+ap['I2']+ap['I3']+bp['I1']+bp['I2']+bp['I3']));assert len(leaves)==7
 def evaluate(a,b,values):
  v=dict(values);v[a['O'][0]]=lut(0x8000,[a['I'+str(i)][0] for i in range(4)],v)
  return lut(0x8000,[b['I'+str(i)][0] for i in range(4)],v)
 checks=0
 for bits in itertools.product((0,1,'x','z'),repeat=7):
  v=dict(zip(leaves,bits));assert evaluate(ap,bp,v)==evaluate(newa,newb,v);checks+=1
 records=[];changed=text
 for n,pins in zip(names,(newa,newb)):
  old=blocks[n];new='\n    EFX_LUT4 '+n+' ('+', '.join('.'+p+'('+render_token(ts[0])+')' for p,ts in pins.items())+') /* exact balanced ready cut */ ;'
  assert changed.count(old)==1;changed=changed.replace(old,new,1);records.append(dict(name=n,before=old,after=new))
 restored=changed
 for r in records:assert restored.count(r['after'])==1;restored=restored.replace(r['after'],r['before'],1)
 assert restored==text
 out.mkdir()
 for f in base.glob('*.sv'):shutil.copy2(f,out/f.name);assert sha(f)==sha(out/f.name)
 (out/'native_source.v').write_text(changed,encoding='utf8')
 rp=out/'ready-balanced-edit.json';rp.write_text(json.dumps(dict(base=str(base),base_sha256=sha(base/'native_source.v'),records=records,checks=checks,boundary=leaves,changed_state=0,latency_added=0,late_fault_lut_levels_before=2,late_fault_lut_levels_after=1,early_other_conditions_extra_level=1),indent=2))
 graft=json.loads((base/'graft.json').read_text());graft['ready_balanced']=dict(base=str(base),manifest=str(rp),manifest_sha256=sha(rp));graft['inputs'][str(Path(__file__).resolve())]=sha(__file__)
 (out/'graft.json').write_text(json.dumps(graft,indent=2))
 inputs={**parent['inputs'],**{str(f):sha(f) for f in (pp,Path(__file__).resolve(),rp,out/'graft.json',out/'native_source.v')}}
 proof={**parent,'ready_balanced_parent':str(base),'ready_balanced_checks':checks,'ready_balanced_reversible_text':True,'inputs':inputs,'hardware_access':False,'board_release':False}
 (out/'native-graph-audit.json').write_text(json.dumps(proof,indent=2))
 print(json.dumps(dict(status=proof['status'],checks=checks,retained=proof['retained_primitives'],source_sha256=sha(out/'native_source.v'))))
if __name__=='__main__':main()
