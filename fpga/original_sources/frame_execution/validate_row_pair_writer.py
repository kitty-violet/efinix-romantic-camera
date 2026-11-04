"""Four-state layout/partial-half/held-source/numeric-fault writer verification."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,MS,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
records=[]
for name in ('row_pair_column_writer.sv','tb_row_pair_column_writer.sv'):
 src=HERE/'variants/row_pair_arena'/name;dst=r/name;shutil.copy2(src,dst);records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(src)))
(r/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2)+'\n')
run([MS/'vlib.exe','work'],r,'vlib.log');body,_=run([MS/'vlog.exe','-sv',*[i['snapshot'] for i in records]],r,'compile.log');assert 'Errors: 0, Warnings: 0' in body
results=[]
for seed in (71,391,1027):
 body,sec=run([MS/'vsim.exe','-c','work.tb_row_pair_column_writer',f'+SEED={seed}','-do','onerror {quit -code 1}; run -all; quit -code 0'],r,f'simulation_{seed}.log',timeout=60)
 lines=re.findall(r'^# (PASS ROW_PAIR_WRITER.*)$',body,re.M);assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
 results.append(dict(seed=seed,line=lines[0],seconds=sec,log_sha256=sha(r/f'simulation_{seed}.log')));print(results[-1],flush=True)
for i in records:assert sha(i['source'])==sha(i['snapshot'])==i['sha256']
(r/'result.json').write_text(json.dumps(dict(status='ROW_PAIR_WRITER_FOUR_STATE_PASS',runs=results,inputs_sha256=sha(r/'inputs.json'),integrated_cnn=False,whole_board=False),indent=2)+'\n')
