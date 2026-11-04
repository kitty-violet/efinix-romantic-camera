"""Exact cycle comparison against the RAM FIFO using official SRL and RAM models."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,EF,MS,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--overrides-dir',type=Path);a=p.parse_args()
root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
paths=[EF/'sim_models/verilog/efx_ram10.v',EF/'sim_models/verilog/efx_srl8.v',HERE/'variants/fifo80/banked_result_fifo.sv',HERE/'shallow_result_fifo.sv',HERE/'tb_shallow_result_fifo.sv'];records=[]
if a.overrides_dir:
 override=a.overrides_dir.resolve();assert override.is_relative_to(HERE/'variants')
 paths=[override/f.name if (override/f.name).is_file() else f for f in paths]
for f in paths:
 dst=root/f.name;shutil.copy2(f,dst);records.append(dict(path=str(f),snapshot=str(dst),sha256=sha(f)))
(root/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2))
run([MS/'vlib.exe','work'],root,'vlib.log')
body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log');assert 'Errors: 0, Warnings: 0' in body
body,seconds=run([MS/'vsim.exe','-c','work.tb_shallow_result_fifo','-do','onerror {quit -code 1}; run -all; quit -code 0'],root,'simulation.log')
lines=re.findall(r'^# (PASS SHALLOW_FIFO.*)$',body,re.M);assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
for r in records:assert sha(r['path'])==sha(r['snapshot'])==r['sha256']
result=dict(status='SHALLOW_FIFO_OFFICIAL_PRIMITIVE_EQUIVALENCE_PASS',line=lines[0],seconds=seconds,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'))
(root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
