"""Isolated mixed-service pool, real DSP/RAM primitives, queue/order/cancel scoreboard."""
import argparse,json,re,shutil,traceback
from run_units import HERE,ROOT,EF,MS,sha,run
from validate_compute_mailbox import selected as base
def selected(compact=False):
 names={p.name for p in __import__('validate_banked_pool').sources()}
 return [HERE/'variants'/('mixed_fifo80' if compact else 'mixed_rq')/p.name if p.name=='shared_pool_banked_fifo.sv' else HERE/'variants/fifo80'/p.name if p.name=='banked_result_fifo.sv' else p for p in base() if p.name in names]
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=__import__('pathlib').Path,required=True);p.add_argument('--compact',action='store_true');a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
 try:
  paths=[EF/'sim_models/verilog'/n for n in ('efx_dsp24.v','efx_ram10.v')]+selected(a.compact)+[HERE/'tb_mixed_pool.sv'];records=[]
  for f in paths:
   dst=root/f.name;shutil.copy2(f,dst);records.append(dict(path=str(f),snapshot=str(dst),sha256=sha(f)))
  (root/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2))
  run([MS/'vlib.exe','work'],root,'vlib.log')
  body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log');assert 'Errors: 0, Warnings: 0' in body
  rows=[]
  for seed in (71,391):
   body,secs=run([MS/'vsim.exe','-c','work.tb_mixed_pool','-gSEED='+str(seed),'-do','onerror {quit -code 1}; run -all; quit -code 0'],root,f'seed{seed}.log')
   lines=re.findall(r'^# (PASS MIXED_POOL.*)$',body,re.M);assert len(lines)==1
   rows.append(dict(seed=seed,line=lines[0],seconds=secs,log_sha256=sha(root/f'seed{seed}.log')))
  for r in records:assert sha(r['path'])==sha(r['snapshot'])==r['sha256']
  result=dict(status='LOCAL_MIXED_RQ_SERVICE_PASS',results=rows,inputs_sha256=sha(root/'inputs.json'),real_pw_residual_pairing=False,full_frame=False,pnr=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()

