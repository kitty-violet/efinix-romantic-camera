"""Four-state packet equivalence under independent per-port stalls and cancellation."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,MS,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--candidate',type=Path,default=HERE/'variants/multi_issue_reader/unified_column_reader.sv');a=p.parse_args()
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp');r.mkdir(parents=True,exist_ok=False)
records=[]
candidate=a.candidate.resolve();assert candidate.is_relative_to(HERE/'variants')
testbench=candidate.parent/'tb_multi_issue_reader.sv'
if not testbench.is_file():testbench=HERE/'tb_multi_issue_reader.sv'
for f in (HERE/'unified_column_reader.sv',candidate,testbench):
 name='reader_reference.sv' if f==HERE/'unified_column_reader.sv' else f.name
 dst=r/name
 if name=='reader_reference.sv':dst.write_text(f.read_text().replace('module unified_column_reader(','module reader_reference('))
 else:shutil.copy2(f,dst)
 records.append(dict(path=str(f),snapshot=str(dst),source_sha256=sha(f),snapshot_sha256=sha(dst)))
shutil.copy2(__file__,r/'frozen_runner.py')
(r/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(r/'frozen_runner.py')),indent=2))
run([MS/'vlib.exe','work'],r,'vlib.log')
body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[x['snapshot'] for x in records]],r,'compile.log')
assert 'Errors: 0, Warnings: 0' in body
body,seconds=run([MS/'vsim.exe','-c','work.tb_multi_issue_reader','-do','onerror {quit -code 1}; run -all; quit -code 0'],r,'simulation.log')
lines=re.findall(r'^# (PASS MULTI_ISSUE_READER.*)$',body,re.M);assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
for x in records:assert sha(x['path'])==x['source_sha256'] and sha(x['snapshot'])==x['snapshot_sha256']
result=dict(status='FOUR_STATE_READER_PACKET_EQUIVALENCE_PASS',line=lines[0],seconds=seconds,inputs_sha256=sha(r/'inputs.json'),log_sha256=sha(r/'simulation.log'),actual_ram=False,full_frame=False)
(r/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
