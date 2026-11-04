"""Exact valid-lane arithmetic, overflow flags and four-cycle return A/B."""
import argparse,json,re,shutil,traceback
from pathlib import Path
from run_units import ROOT,HERE,MS,EF,run,sha
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
 records=[]
 try:
  srcs=[HERE/'variants/reject_overflow/shared_rq_oc8_cluster.sv',HERE/'variants/reject_overflow/tb_reject_overflow.sv',HERE/'variants/dsp_quant_add/requantize_i32_mul18_q17_pipeline.sv',EF/'sim_models/verilog/efx_dsp48.v',EF/'sim_models/verilog/efx_srl8.v']
  for s in srcs:
   d=root/s.name;shutil.copy2(s,d);records.append(dict(path=str(s),snapshot=str(d),sha256=sha(s)))
  (root/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2))
  run([MS/'vlib.exe','work'],root,'vlib.log')
  body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log');assert 'Errors: 0, Warnings: 0' in body
  body,seconds=run([MS/'vsim.exe','-c','work.tb_reject_overflow','-do','onerror {quit -code 1}; run -all; quit -code 0'],root,'simulation.log',timeout=120)
  lines=re.findall(r'^# (PASS REJECT_OVERFLOW.*)$',body,re.M);assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
  for r in records:assert sha(r['path'])==sha(r['snapshot'])==r['sha256']
  result=dict(status='REJECT_OVERFLOW_FOUR_STATE_PASS',line=lines[0],seconds=seconds,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),same_latency=4,invalid_bias_payload_unspecified=True,full_frame=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
