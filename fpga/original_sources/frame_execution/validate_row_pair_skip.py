"""Four-state residual routing/signed-sum/retained-return short gate."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,MS,EF,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int,action='append');a=p.parse_args()
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
v=HERE/'variants/row_pair_arena';records=[]
for src in [EF/'sim_models/verilog/efx_srl8.v',ROOT/'tmp/frame-execution/connected-board-compact-feed1/assembly/shallow_result_fifo.sv',*[v/n for n in ('row_pair_skip_fuse4.sv','tb_row_pair_skip_fuse4.sv')]]:
 dst=r/src.name;shutil.copy2(src,dst);records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(src)))
(r/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2)+'\n')
run([MS/'vlib.exe','work'],r,'vlib.log');body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[i['snapshot'] for i in records]],r,'compile.log');assert 'Errors: 0, Warnings: 0' in body
results=[]
for seed in (a.seed or [71,391,1027]):
 body,sec=run([MS/'vsim.exe','-c','work.tb_row_pair_skip_fuse4',f'+SEED={seed}','-do','onerror {quit -code 1}; run -all; quit -code 0'],r,f'simulation_{seed}.log',timeout=90)
 lines=re.findall(r'^# (PASS ROW_PAIR_SKIP.*)$',body,re.M);assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
 results.append(dict(seed=seed,line=lines[0],seconds=sec,log_sha256=sha(r/f'simulation_{seed}.log')));print(results[-1],flush=True)
for i in records:assert sha(i['source'])==sha(i['snapshot'])==i['sha256']
(r/'result.json').write_text(json.dumps(dict(status='ROW_PAIR_SKIP_FOUR_STATE_PASS',runs=results,inputs_sha256=sha(r/'inputs.json'),requantization='identity behavioral feedback; production RQ integration still open',integrated_cnn=False,whole_board=False),indent=2)+'\n')
