"""Official SRL concurrent active/staged groups, all 16 phase-count pairs."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,EF,MS,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--variant',type=Path);p.add_argument('--testbench',type=Path);a=p.parse_args()
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
v=a.variant.resolve() if a.variant else HERE/'variants/fixed_prefetch'
tb=a.testbench.resolve() if a.testbench else HERE/'variants/fixed_prefetch_test/tb_fixed_prefetch_weights.sv'
assert v.is_relative_to(ROOT) and tb.is_relative_to(ROOT)
files=[v/'fixed_ring_weights9.sv',tb,EF/'sim_models/verilog/efx_srl8.v'];records=[]
for f in files:
 d=r/f.name;shutil.copy2(f,d);records.append(dict(source=str(f),snapshot=str(d),sha256=sha(f)))
(r/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2)+'\n')
run([MS/'vlib.exe','work'],r,'vlib.log')
b,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[x['snapshot'] for x in records]],r,'compile.log');assert 'Errors: 0, Warnings: 0' in b
b,sec=run([MS/'vsim.exe','-c','work.tb_fixed_prefetch_weights','-do','onerror {quit -code 1}; run -all; quit -code 0'],r,'simulation.log',timeout=90)
lines=re.findall(r'^# (PASS FIXED_PREFETCH_WEIGHTS.*)$',b,re.M);assert len(lines)==1 and 'Errors: 0, Warnings: 0' in b
for x in records:assert sha(x['source'])==sha(x['snapshot'])==x['sha256']
result=dict(status='FIXED_PREFETCH_WEIGHTS_FOUR_STATE_PASS',line=lines[0],seconds=sec,inputs_sha256=sha(r/'inputs.json'),log_sha256=sha(r/'simulation.log'))
(r/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
