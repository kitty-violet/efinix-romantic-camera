"""Actual lookup -> finite dispatcher -> ONE selected 32/4 pool regression."""
import argparse,json,re,shutil,traceback
from pathlib import Path
from run_units import HERE,ROOT,EF,MS,sha,run
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seeds',type=int,nargs='+',default=[71,391,1027]);a=p.parse_args()
 build=a.root.resolve();assert build.is_relative_to(ROOT/'tmp');build.mkdir(parents=True,exist_ok=False)
 fw=HERE.parent/'feature_window';selection=json.loads((fw/'shared_pool_selected_sources.json').read_text())
 paths=[EF/'sim_models/verilog/efx_dsp24.v',EF/'sim_models/verilog/efx_dpram10.v']
 for row in selection['sources']:
  f=ROOT/row['path'];assert sha(f).upper()==row['sha256']
  paths.append(HERE.parent/'pw_array_feed_compact'/f.name if f.name=='requantize_i32_mul18_q17_pipeline.sv' else f)
 paths += [HERE/'residual_lookup_align32.sv',HERE/'residual_pool_dispatch32.sv',HERE/'tb_residual_pool.sv']
 records=[];tables=ROOT/'tmp/frame-execution/residual-tables1'
 try:
  for f in paths+[Path(__file__),HERE/'residual_tables.py',ROOT/'network_ab/model_b_contract.json',ROOT/'network_ab/contract_reference.py',tables/'table.hex',tables/'crc.hex']:
   dst=build/f.name;assert not dst.exists();shutil.copy2(f,dst);records.append(dict(path=str(f),snapshot=str(dst),sha256=sha(f)))
  (build/'inputs.json').write_text(json.dumps(dict(inputs=records,fronts=32,clusters=4),indent=2))
  run([MS/'vlib.exe','work'],build,'vlib.log')
  body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[build/f.name for f in paths]],build,'compile.log')
  assert 'Errors: 0, Warnings: 0' in body
  rows=[]
  for seed in a.seeds:
   body,secs=run([MS/'vsim.exe','-c','work.tb_residual_pool','-gSEED='+str(seed),'+VECTORS='+build.as_posix(),
     '-do','onerror {quit -code 1}; run -all; quit -code 0'],build,f'seed{seed}.log')
   lines=re.findall(r'^# ((?:PASS|MEASURE) RESIDUAL_POOL.*)$',body,re.M)
   assert len(lines)==2 and 'Errors: 0, Warnings: 0' in body
   rows.append(dict(seed=seed,lines=lines,seconds=secs,log_sha256=sha(build/f'seed{seed}.log')))
  for row in records:assert sha(row['path'])==sha(row['snapshot'])==row['sha256']
  result=dict(status='LOCAL_RESIDUAL_LOOKUP_SINGLE_SHARED_POOL_PASS',results=rows,inputs_sha256=sha(build/'inputs.json'),
   full_frame=False,pnr=False,real_upstream_memory=False,trained_parameters=False)
  (build/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 except Exception as e:
  (build/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
