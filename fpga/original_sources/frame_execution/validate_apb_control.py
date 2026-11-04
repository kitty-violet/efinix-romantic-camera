"""APB coefficient load/readback against actual official RAM/store."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import HERE,ROOT,EF,MS,sha,run
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
paths=[EF/'sim_models/verilog/efx_ram10.v',ROOT/'tmp/frame-execution/program-map2/resident_weight_bank64.sv',
 ROOT/'tmp/frame-execution/program-map2/coefficient_bank_store_fast.sv',HERE/'cnn_apb_control.sv',HERE/'tb_cnn_apb_control.sv']
records=[]
for src in paths:
 dst=root/src.name;shutil.copy2(src,dst);records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
(root/'inputs.json').write_text(json.dumps(dict(inputs=records),indent=2))
run([MS/'vlib.exe','work'],root,'vlib.log')
body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log');assert 'Errors: 0, Warnings: 0' in body
body,secs=run([MS/'vsim.exe','-c','work.tb_cnn_apb_control','+VECTORS='+str(ROOT/'tmp/frame-execution/joint-reorder-vectors1'),
 '-do','onerror {quit -code 1}; run -all; quit -code 0'],root,'simulation.log',timeout=120)
lines=re.findall(r'^# (PASS CNN_APB_CONTROL.*)$',body,re.M);assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
for r in records:assert sha(r['source'])==sha(r['snapshot'])==r['sha256']
result=dict(status='CNN_APB_CONTROL_FOURSTATE_PASS',lines=lines,seconds=secs,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),cpu_firmware_test=False)
(root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

