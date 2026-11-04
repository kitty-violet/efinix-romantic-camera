"""Actual address RTL against independently enumerated taps, all source modes."""
import argparse,json,re,shutil,traceback
from pathlib import Path
from run_units import HERE,ROOT,MS,sha,run
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 build=a.root.resolve();assert build.is_relative_to(ROOT/'tmp');build.mkdir(parents=True,exist_ok=False)
 paths=[HERE/'dw_phase_plan32.sv',HERE/'tb_dw_phase_plan32.sv',Path(__file__)];records=[]
 try:
  for f in paths:
   dst=build/f.name;shutil.copy2(f,dst);records.append(dict(path=str(f),snapshot=str(dst),sha256=sha(f)))
  (build/'inputs.json').write_text(json.dumps(dict(inputs=records),indent=2))
  run([MS/'vlib.exe','work'],build,'vlib.log')
  body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[build/f.name for f in paths[:2]]],build,'compile.log')
  assert 'Errors: 0, Warnings: 0' in body
  rows=[]
  for seed in (71,391,1027):
   body,secs=run([MS/'vsim.exe','-c','work.tb_dw_phase_plan32','-gSEED='+str(seed),'-do','onerror {quit -code 1}; run -all; quit -code 0'],build,f'seed{seed}.log')
   lines=re.findall(r'^# ((?:PASS DW32_PLAN|MEASURE DW32).*)$',body,re.M)
   assert len(lines)==4 and 'Errors: 0, Warnings: 0' in body
   rows.append(dict(seed=seed,lines=lines,seconds=secs,log_sha256=sha(build/f'seed{seed}.log')))
  for r in records:assert sha(r['path'])==sha(r['snapshot'])==r['sha256']
  result=dict(status='LOCAL_DW32_ADDRESS_PHASE_SOURCE_PASS',results=rows,inputs_sha256=sha(build/'inputs.json'),
   actual_memory_reads=False,result_write_arbitration=False,complete_DW_execution=False,full_frame=False,pnr=False)
  (build/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 except Exception as e:
  (build/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
