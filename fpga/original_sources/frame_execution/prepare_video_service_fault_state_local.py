"""Two equivalent fault state copies dedicated to the slow global reductions.

Both copies use exactly the original D/CE/SR/CLK and parameters. They do not
add latency or change fault semantics. No ordinary physical location is fixed.
"""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,sha
from native_local_drivers import CELL,graph,render_token
from probe_full_native_split_map import params

def main():
 p=argparse.ArgumentParser()
 for n in ('base','root'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();base=a.base.resolve();out=a.root.resolve()
 assert out.is_relative_to(ROOT/'tmp') and not out.exists()
 pp=base/'native-graph-audit.json';proof=json.loads(pp.read_text())
 assert proof['status']=='SERVICE_NATIVE_RETAINED_GRAPH_AND_CUT_PASS'
 for f,h in proof['inputs'].items():assert sha(f)==h,f
 text=(base/'native_source.v').read_text();prefix='video_processing_pipeline/pipeline/cnn/core/dut/engine/'
 plan=[(prefix+'mac/werror~FF','LUT__132940','I3'),(prefix+'rq_error~FF','LUT__132941','I0')]
 names={n for old,load,pin in plan for n in (old,load)}
 blocks={m[2].lstrip('\\'):m[0] for m in CELL.finditer(text) if m[2].lstrip('\\') in names}
 assert set(blocks)==names
 g=graph('\n'.join(blocks.values()));ps=params(text)
 changed=text;insert='';declaration='';records=[]
 for i,(old,load,pin) in enumerate(plan):
  k,pins=g[old];assert k=='EFX_FF' and g[load][0]=='EFX_LUT4'
  assert pins['CLK']==[('net','process_clk_150',None)]
  assert g[load][1][pin]==pins['Q']
  newname=f'SERVICE_LOCAL_FAULT_STATE_{i}';net=('net',f'service_local_fault_state_{i}',None)
  assert newname not in text and net[1] not in text
  newpins={**pins,'Q':[net]}
  cell='\n    EFX_FF '+newname+' ('+', '.join('.'+p+'('+render_token(ts[0])+')' for p,ts in newpins.items())+') /* exact fault-state copy */ ;\n'
  assert graph(cell)[newname]==('EFX_FF',newpins)
  parameter_lines=''.join(f"    defparam {newname}.{p} = {w}'d{v};\n" for p,(w,v) in ps[old].items())
  assert params(parameter_lines)[newname]==ps[old]
  assert ps[old].get('INIT',(1,0))==(1,0)
  # Identical initial state and identical complete transition inputs imply
  # equality at every clock, including X-valued D/CE/SR in the vendor model.
  assert {p:v for p,v in newpins.items() if p!='Q'}=={p:v for p,v in pins.items() if p!='Q'}
  before=blocks[load];pattern=r'\.'+pin+r'\([^()]*\)';assert len(re.findall(pattern,before))==1
  after=re.sub(pattern,lambda m:'.'+pin+'('+render_token(net)+')',before)
  assert graph(after)[load]==('EFX_LUT4',{**g[load][1],pin:[net]})
  assert changed.count(before)==1;changed=changed.replace(before,after,1)
  insert+=cell+parameter_lines
  declaration+=f'    (* syn_keep="true" *) wire {net[1]};\n'
  records.append(dict(original=old,copy=newname,parameters=ps[old],original_pins=pins,copy_pins=newpins,load=load,pin=pin,before=before,after=after))
 pos=CELL.search(changed).start();changed=changed[:pos]+declaration+changed[pos:]
 pos=changed.index('    protected_cell protected_inst (');changed=changed[:pos]+insert+changed[pos:]
 restored=changed.replace(declaration,'',1).replace(insert,'',1)
 for r in records:restored=restored.replace(r['after'],r['before'],1)
 assert restored==text
 out.mkdir()
 for f in base.glob('*.sv'):shutil.copy2(f,out/f.name);assert sha(f)==sha(out/f.name)
 (out/'native_source.v').write_text(changed,encoding='utf8')
 receipt=out/'fault-state-edit.json';receipt.write_text(json.dumps(dict(base=str(base),base_sha256=sha(base/'native_source.v'),records=records,insert=insert,declaration=declaration,induction='Same INIT and exact D/CE/SR/CLK/parameters; no additional cycle',ordinary_fixed_locations_added=0),indent=2))
 graft=json.loads((base/'graft.json').read_text());graft['fault_state_local']=dict(base=str(base),manifest=str(receipt),manifest_sha256=sha(receipt));graft['inputs'][str(Path(__file__).resolve())]=sha(__file__)
 (out/'graft.json').write_text(json.dumps(graft,indent=2))
 inputs={**proof['inputs'],**{str(f):sha(f) for f in (pp,receipt,Path(__file__).resolve(),out/'native_source.v',out/'graft.json')}}
 result={**proof,'retained_primitives':proof['retained_primitives']+2,'fault_state_parent':str(base),'fault_state_copies':2,'fault_state_reversible_text':True,'inputs':inputs,'hardware_access':False,'board_release':False}
 (out/'native-graph-audit.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(dict(status=result['status'],source_sha256=sha(out/'native_source.v'),retained=result['retained_primitives'],copies=2,latency_added=0)))
if __name__=='__main__':main()
