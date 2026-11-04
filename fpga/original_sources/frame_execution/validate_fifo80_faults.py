"""Production compact queue mask/error rejection using real DSP and RAM."""
import argparse,json,re,shutil,traceback
from pathlib import Path
from run_units import HERE,ROOT,EF,MS,sha,run
from validate_compute_fifo80 import selected
from validate_banked_pool import sources
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--mixed-compact',action='store_true');a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
 try:
  names={f.name for f in sources()};paths=[EF/'sim_models/verilog'/n for n in ('efx_dsp24.v','efx_ram10.v')]+[f for f in selected() if f.name in names]+[HERE/'tb_fifo80_faults.sv'];records=[]
  if a.mixed_compact:paths=[HERE/'variants/mixed_fifo80'/f.name if f.name=='shared_pool_banked_fifo.sv' else f for f in paths]
  for f in paths:
   dst=root/f.name;shutil.copy2(f,dst);records.append(dict(path=str(f),snapshot=str(dst),sha256=sha(f)))
  (root/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2))
  run([MS/'vlib.exe','work'],root,'vlib.log')
  defines=['+define+MIXED_COMPACT'] if a.mixed_compact else []
  body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*defines,*[r['snapshot'] for r in records]],root,'compile.log');assert 'Errors: 0, Warnings: 0' in body
  body,secs=run([MS/'vsim.exe','-c','work.tb_fifo80_faults','-do','onerror {quit -code 1}; run -all; quit -code 0'],root,'simulation.log')
  lines=re.findall(r'^# (PASS FIFO80_FAULTS.*)$',body,re.M);assert len(lines)==1
  for r in records:assert sha(r['path'])==sha(r['snapshot'])==r['sha256']
  result=dict(status='LOCAL_FIFO80_POISON_AND_RECOVERY_PASS',line=lines[0],seconds=secs,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),full_frame=False,pnr=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()

