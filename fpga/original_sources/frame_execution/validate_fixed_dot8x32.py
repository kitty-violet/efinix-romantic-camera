"""Official four-state arithmetic gate, no board/supply/performance claim."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,EF,MS,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
v=HERE/'variants/fixed_dot8x32';records=[]
for f in [v/'fixed_dot8x32.sv',v/'tb_fixed_dot8x32.sv',EF/'sim_models/verilog/efx_dsp24.v',EF/'sim_models/verilog/efx_srl8.v']:
 dst=r/f.name;shutil.copy2(f,dst);records.append(dict(source=str(f),snapshot=str(dst),sha256=sha(f)))
(r/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2)+'\n')
run([MS/'vlib.exe','work'],r,'vlib.log');body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[x['snapshot'] for x in records]],r,'compile.log');assert 'Errors: 0, Warnings: 0' in body
results=[]
for seed in (71,391,1027):
 body,seconds=run([MS/'vsim.exe','-c','work.tb_fixed_dot8x32',f'+SEED={seed}','-do','onerror {quit -code 1}; run -all; quit -code 0'],r,f'simulation_{seed}.log',timeout=120)
 lines=re.findall(r'^# (PASS FIXED_DOT8X32.*)$',body,re.M);assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
 results.append(dict(seed=seed,line=lines[0],seconds=seconds,log_sha256=sha(r/f'simulation_{seed}.log')));print(results[-1],flush=True)
for x in records:assert sha(x['source'])==sha(x['snapshot'])==x['sha256']
(r/'result.json').write_text(json.dumps(dict(status='FIXED_DOT8X32_OFFICIAL_ARITHMETIC_PASS',runs=results,inputs_sha256=sha(r/'inputs.json'),real_supply=False,whole_cnn=False,physical=False),indent=2)+'\n')
