"""Guarded-front exact cycle equivalence of direct feedback and prefix subtraction."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,EF,MS,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--overrides-dir',type=Path);a=p.parse_args()
root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
paths=[EF/'sim_models/verilog/efx_dsp24.v',HERE.parent/'cnn_microbench/int8_dual_mac_lane_dspacc.sv',HERE/'variants/direct_accum/int8_dual_mac_lane_prefix_stream.sv',HERE/'tb_direct_accum.sv'];records=[]
if a.overrides_dir:
 override=a.overrides_dir.resolve();assert override.is_relative_to(HERE/'variants')
 paths=[override/f.name if (override/f.name).is_file() else f for f in paths]
for f in paths:
 dst=root/f.name;shutil.copy2(f,dst);records.append(dict(path=str(f),snapshot=str(dst),sha256=sha(f)))
ref=HERE/'variants/compact_quant/int8_dual_mac_lane_prefix_stream.sv';dst=root/'reference_prefix.sv'
dst.write_text(ref.read_text().replace('module int8_dual_mac_lane_prefix_stream','module reference_prefix'))
records.append(dict(path=str(ref),snapshot=str(dst),sha256=sha(dst),source_sha256=sha(ref),transform='module rename only'))
(root/'inputs.json').write_text(json.dumps(dict(inputs=records),indent=2))
run([MS/'vlib.exe','work'],root,'vlib.log')
body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log');assert 'Errors: 0, Warnings: 0' in body
body,seconds=run([MS/'vsim.exe','-c','work.tb_direct_accum','-do','onerror {quit -code 1}; run -all; quit -code 0'],root,'simulation.log')
lines=re.findall(r'^# (PASS DIRECT_ACCUM.*)$',body,re.M);assert len(lines)==1
for r in records:assert sha(r['snapshot'])==r['sha256'] and sha(r['path'])==r.get('source_sha256',r['sha256'])
result=dict(status='DIRECT_ACCUM_LEGAL_FRONT_EQUIVALENCE_PASS',line=lines[0],seconds=seconds,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),adopted=False)
(root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
