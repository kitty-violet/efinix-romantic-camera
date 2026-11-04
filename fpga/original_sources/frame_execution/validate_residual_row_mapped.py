"""Official mapped row controller + actual source pool and physical RAM models.
Uses public interfaces only. Full-row oracle and injected failures retained.
"""
import argparse,json,re,shutil,traceback
from pathlib import Path
from run_units import HERE,ROOT,MS,EF,sha,run
from validate_banked_pool import sources
from mapped_pin_audit import audit
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--map',type=Path,required=True)
 p.add_argument('--vectors-from',type=Path,required=True);p.add_argument('--seeds',nargs='+',type=int,default=[71]);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
 records=[]
 try:
  mr=json.loads((a.map/'result.json').read_text());assert mr['status']=='LOCAL_RESIDUAL_ROW_ADAPTER_MAP_PASS'
  mi=json.loads((a.map/'inputs.json').read_text());assert sha(a.map/'inputs.json')==mr['inputs_sha256']
  for r in mi['inputs']:assert sha(r['path'])==sha(r['snapshot'])==r['sha256']
  net=a.map/'outflow/residual_row_shared.map.v';assert sha(net)==mr['map_sha256']
  vr=json.loads((a.vectors_from/'result.json').read_text());vi=json.loads((a.vectors_from/'inputs.json').read_text())
  assert sha(a.vectors_from/'inputs.json')==vr['inputs_sha256']
  shutil.copytree(a.vectors_from/'vectors',root/'vectors')
  for n,h in vi['vectors'].items():assert sha(root/'vectors'/n)==h
  types=set(re.findall(r'^\s*EFX_([A-Z0-9_]+)\s+\S+',net.read_text(),re.M)) | {'DSP24','RAM10','DPRAM10'}
  paths=[EF/'sim_models/verilog'/('efx_'+x.lower()+'.v') for x in sorted(types)]+sources()+[HERE/'tensor_arena12.sv',net,HERE/'tb_residual_row_mapped.sv']
  for f in paths+[Path(__file__),HERE/'mapped_pin_audit.py',HERE/'run_units.py']:
   dst=root/f.name;shutil.copy2(f,dst);records.append(dict(path=str(f),snapshot=str(dst),sha256=sha(f)))
  manifest=dict(inputs=records,map_result_sha256=sha(a.map/'result.json'),vectors_result_sha256=sha(a.vectors_from/'result.json'),oracle=vr['oracle'])
  (root/'inputs.json').write_text(json.dumps(manifest,indent=2))
  run([MS/'vlib.exe','work'],root,'vlib.log')
  body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[root/f.name for f in paths]],root,'compile.log')
  assert 'Errors: 0, Warnings: 0' in body
  results=[]
  for seed in a.seeds:
   body,seconds=run([MS/'vsim.exe','-c','work.tb_residual_row_shared','-gSEED='+str(seed),
    '+VECTORS='+(root/'vectors').as_posix(),'-do','onerror {quit -code 1}; run -all; quit -code 0'],root,f'seed{seed}.log')
   lines=re.findall(r'^# (PASS RESIDUAL_ROW.*)$',body,re.M);assert len(lines)==1
   rows=re.findall(r'^# ROW (.*)$',body,re.M);assert len(rows)==vr['oracle']['cases']+1
   warnings=audit(body,net.read_text())
   results.append(dict(seed=seed,line=lines[0],rows=rows,seconds=seconds,warnings=warnings,log_sha256=sha(root/f'seed{seed}.log')))
  for r in records:assert sha(r['path'])==sha(r['snapshot'])==r['sha256']
  result=dict(status='LOCAL_RESIDUAL_ROW_OFFICIAL_MAP_NUMERIC_PASS',results=results,map_sha256=sha(net),inputs_sha256=sha(root/'inputs.json'),
    full_frame=False,pnr=False,production_parameters=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()

