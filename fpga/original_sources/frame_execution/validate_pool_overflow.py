"""Four-state production pool: all four banks reject overflow before RAM push."""
import argparse,json,re,shutil,traceback
from pathlib import Path
from run_units import ROOT,HERE,MS,run,sha
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--cluster',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
 names={'efx_dsp48.v','efx_dsp24.v','efx_ram10.v','efx_srl8.v','int8_dual_mac_lane_dspacc.sv','int8_dual_mac_lane_prefix_stream.sv','requantize_i32_mul18_q17_pipeline.sv','shared_rq_oc8_cluster.sv','banked_result_fifo.sv','shared_mac_stream_front.sv','shared_pool_banked_fifo.sv','shallow_result_fifo.sv'}
 records=[]
 try:
  cluster=a.cluster.resolve();native=json.loads((cluster/'native_inputs.json').read_text());assert sha(cluster/'frozen/inputs.json')==native['source_manifest_sha256']
  meta=json.loads((cluster/'frozen/inputs.json').read_text());paths=[]
  for r in meta['inputs']:
   src=Path(r['snapshot'])
   if src.name in names:assert sha(src)==r['sha256'];paths.append(src)
  assert {p.name for p in paths}==names
  paths.append(HERE/'variants/reject_overflow/tb_pool_overflow.sv')
  for src in paths:
   dst=root/src.name;shutil.copy2(src,dst);records.append(dict(path=str(src),snapshot=str(dst),sha256=sha(src)))
  (root/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2))
  run([MS/'vlib.exe','work'],root,'vlib.log')
  body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log');assert 'Errors: 0, Warnings: 0' in body
  body,seconds=run([MS/'vsim.exe','-c','work.tb_pool_overflow','-do','onerror {quit -code 1}; run -all; quit -code 0'],root,'simulation.log',timeout=120)
  lines=re.findall(r'^# (PASS POOL_OVERFLOW.*)$',body,re.M);assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body and not re.search(r'\*\* (Error|Fatal)',body)
  for r in records:assert sha(r['path'])==sha(r['snapshot'])==r['sha256']
  result=dict(status='POOL_OVERFLOW_FOUR_STATE_PASS',line=lines[0],seconds=seconds,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),full_frame=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
