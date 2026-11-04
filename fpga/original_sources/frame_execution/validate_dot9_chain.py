"""Bounded official primitive simulation; no whole-CNN feasibility claim."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,EF,MS,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
v=HERE/'variants/fixed_dot9';records=[]
for f in [v/'dsp_dot9_pair_chain.sv',v/'tb_dsp_dot9_pair_chain.sv',EF/'sim_models/verilog/efx_dsp24.v',EF/'sim_models/verilog/efx_srl8.v']:
 dst=r/f.name;shutil.copy2(f,dst);records.append(dict(source=str(f),snapshot=str(dst),sha256=sha(f)))
(r/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2)+'\n')
run([MS/'vlib.exe','work'],r,'vlib.log');body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[x['snapshot'] for x in records]],r,'compile.log');assert 'Errors: 0, Warnings: 0' in body
results=[]
for seed in (71,391,1027):
 body,seconds=run([MS/'vsim.exe','-c','work.tb_dsp_dot9_pair_chain',f'+SEED={seed}','-do','onerror {quit -code 1}; run -all; quit -code 0'],r,f'simulation_{seed}.log',timeout=90)
 lines=re.findall(r'^# (PASS DOT9_CHAIN.*)$',body,re.M);assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
 results.append(dict(seed=seed,line=lines[0],seconds=seconds,log_sha256=sha(r/f'simulation_{seed}.log')))
for x in records:assert sha(x['source'])==sha(x['snapshot'])==x['sha256']
result=dict(status='DOT9_CHAIN_OFFICIAL_NUMERIC_PASS_NOT_PHYSICAL',runs=results,inputs_sha256=sha(r/'inputs.json'),physical_pairing=False,whole_cnn=False)
(r/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
