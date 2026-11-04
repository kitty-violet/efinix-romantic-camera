"""Real-arena scatter checks with independently enumerated convolution taps."""
import argparse,json,random,re,shutil,traceback
from pathlib import Path
from run_units import HERE,ROOT,MS,EF,sha,run
from validate_stride_tap_scatter import vectors
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--bank-lock',action='store_true');p.add_argument('--compact',action='store_true');a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
 try:
  oracle=vectors(root/'vectors')
  files=[EF/'sim_models/verilog/efx_dpram10.v',HERE/'tensor_arena12.sv',HERE/'stride_tap_scatter4.sv',HERE/'tb_stride_tap_scatter4.sv'];records=[]
  if a.bank_lock:files=[HERE/'variants/scatter_bank_lock'/f.name if f.name=='stride_tap_scatter4.sv' else f for f in files]
  if a.compact:files=[HERE/'variants/scatter_compact'/f.name if f.name=='stride_tap_scatter4.sv' else f for f in files]
  for f in files:shutil.copy2(f,root/f.name);records.append(dict(path=str(f),snapshot=str(root/f.name),sha256=sha(f)))
  m=dict(inputs=records,oracle=oracle,dependencies={str(f):sha(f) for f in (Path(__file__),HERE/'stride_tap_layout.py',ROOT/'network_ab/model_b_contract.json')},
   vectors={f.name:sha(f) for f in (root/'vectors').iterdir()})
  (root/'inputs.json').write_text(json.dumps(m,indent=2))
  run([MS/'vlib.exe','work'],root,'vlib.log')
  body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log');assert 'Errors: 0, Warnings: 0' in body
  body,secs=run([MS/'vsim.exe','-c','work.tb_stride_tap_scatter4','+VECTORS='+(root/'vectors').as_posix(),'-do','onerror {quit -code 1}; run -all; quit -code 0'],root,'simulation.log')
  lines=re.findall(r'^# (PASS STRIDE_TAP_SCATTER4.*)$',body,re.M);rows=re.findall(r'^# SCATTER4 (.*)$',body,re.M);assert len(lines)==1 and len(rows)==oracle['cases']*2+5
  for r in records:assert sha(r['path'])==sha(r['snapshot'])==r['sha256']
  result=dict(status='LOCAL_FOUR_INGRESS_REAL_ARENA_TAP_SCATTER_PASS',line=lines[0],rows=rows,seconds=secs,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),producer_rq_connected=False,real_stride_compute_connected=False,full_frame=False,pnr=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
