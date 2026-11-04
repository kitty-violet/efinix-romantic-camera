"""Four-state prefix refactor comparison at extrema, wrap, stalls and cancellation."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,EF,MS,sha,run
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
base=ROOT/'tmp/frame-execution/byte-rotated-map1'
candidate=HERE/'variants/reuse_integrated'
records=[]
for name in ('int8_dual_mac_lane_dspacc.sv',):
 src=base/name;dst=root/name;shutil.copy2(src,dst);records.append(dict(path=str(src),snapshot=str(dst),sha256=sha(dst)))
for name in ('shared_mac_stream_front.sv','int8_dual_mac_lane_prefix_stream.sv'):
 src=base/name;dst=root/('reference_'+name)
 body=src.read_text().replace('shared_mac_stream_front','reference_front').replace('int8_dual_mac_lane_prefix_stream','reference_prefix')
 dst.write_text(body);records.append(dict(path=str(src),snapshot=str(dst),sha256=sha(dst),source_sha256=sha(src),rename_only=True))
 src=candidate/name;dst=root/name;shutil.copy2(src,dst);records.append(dict(path=str(src),snapshot=str(dst),sha256=sha(dst)))
for src in (EF/'sim_models/verilog/efx_dsp24.v',HERE/'variants/shared_prefix/tb_shared_prefix_wrap.sv'):
 dst=root/src.name;shutil.copy2(src,dst);records.append(dict(path=str(src),snapshot=str(dst),sha256=sha(dst)))
(root/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2))
run([MS/'vlib.exe','work'],root,'vlib.log')
body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log')
assert 'Errors: 0, Warnings: 0' in body
body,seconds=run([MS/'vsim.exe','-c','work.tb_shared_prefix_wrap','-do','onerror {quit -code 1}; run -all; quit -code 0'],root,'simulation.log',timeout=180)
lines=re.findall(r'^# (PASS SHARED_PREFIX_WRAP.*)$',body,re.M);assert len(lines)==1
for r in records:assert sha(Path(r['snapshot']))==r['sha256']
result=dict(status='SHARED_PREFIX_EXTREMA_WRAP_FOURSTATE_PASS',line=lines[0],seconds=seconds,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'))
(root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
