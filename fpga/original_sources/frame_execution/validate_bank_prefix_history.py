"""Compare a shared eight-entry bank prefix history to the frozen private copies."""
import argparse,json,re,shutil,traceback
from pathlib import Path
from run_units import ROOT,HERE,MS,EF,run,sha
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
 base=ROOT/'tmp/frame-execution/connected-board-shared-taps1/assembly'
 variant=HERE/'variants/bank_prefix_history';records=[]
 try:
  ref=base/'shared_rq_oc8_cluster.sv';dst=root/'rq_history_reference.sv'
  dst.write_text(ref.read_text().replace('module shared_rq_oc8_cluster #(','module rq_history_reference #('))
  records.append(dict(path=str(ref),snapshot=str(dst),source_sha256=sha(ref),sha256=sha(dst),rename_only=True))
  for src in (base/'requantize_i32_mul18_q17_pipeline.sv',variant/'shared_rq_oc8_cluster.sv',variant/'tb_bank_prefix_history.sv',EF/'sim_models/verilog/efx_srl8.v'):
   dst=root/src.name;shutil.copy2(src,dst);records.append(dict(path=str(src),snapshot=str(dst),source_sha256=sha(src),sha256=sha(dst)))
  (root/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2))
  run([MS/'vlib.exe','work'],root,'vlib.log')
  body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log')
  assert 'Errors: 0, Warnings: 0' in body
  body,seconds=run([MS/'vsim.exe','-c','work.tb_bank_prefix_history','-do','onerror {quit -code 1}; run -all; quit -code 0'],root,'simulation.log',timeout=180)
  lines=re.findall(r'^# (PASS BANK_PREFIX_HISTORY.*)$',body,re.M)
  assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
  for row in records:assert sha(row['snapshot'])==row['sha256'] and sha(row['path'])==row['source_sha256']
  result=dict(status='BANK_PREFIX_HISTORY_FOUR_STATE_EQUIVALENCE_PASS',line=lines[0],seconds=seconds,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),full_frame=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
