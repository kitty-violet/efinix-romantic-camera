"""Exhaustive INT8 product check before the official pipelined arithmetic gate."""
import argparse,json,shutil
from pathlib import Path
from run_units import ROOT,HERE,MS,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
records=[]
for f in [HERE/'variants/fixed_tail_booth/fixed_window_dot8x32.sv',HERE/'variants/fixed_tail_booth_test/tb_tail_mul8.sv']:
 d=r/f.name;shutil.copy2(f,d);records.append(dict(source=str(f),snapshot=str(d),sha256=sha(f)))
(r/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2))
run([MS/'vlib.exe','work'],r,'vlib.log')
body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[x['snapshot'] for x in records]],r,'compile.log')
assert 'Errors: 0, Warnings: 0' in body
body,seconds=run([MS/'vsim.exe','-c','work.tb_tail_mul8','-do','onerror {quit -code 1}; run -all; quit -code 0'],r,'simulation.log',timeout=120)
assert 'PASS TAIL_MUL8_EXHAUSTIVE checks=65536' in body and 'Errors: 0, Warnings: 0' in body
for x in records:assert sha(x['source'])==sha(x['snapshot'])==x['sha256']
result=dict(status='TAIL_MUL8_EXHAUSTIVE_PASS',checks=65536,seconds=seconds,inputs_sha256=sha(r/'inputs.json'),log_sha256=sha(r/'simulation.log'))
(r/'result.json').write_text(json.dumps(result,indent=2));print(result)
