"""Four-state reject-before-access checks using the exact verified candidate."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,MS,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);a=p.parse_args()
root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
ref=a.reference.resolve();r=json.loads((ref/'result.json').read_text());assert 'PASS' in r['status']
assert sha(ref/'inputs.json')==r['inputs_sha256'];source=json.loads((ref/'inputs.json').read_text())
records=[]
for row in source['inputs']:
 src=Path(row['snapshot']);assert sha(src)==row['sha256']
 if src.name.startswith('tb_'):continue
 dst=root/src.name;shutil.copy2(src,dst);records.append(dict(snapshot=str(dst),sha256=sha(dst)))
tb=HERE/'tb_specialized_rejection.sv';dst=root/tb.name;shutil.copy2(tb,dst);records.append(dict(snapshot=str(dst),sha256=sha(dst)))
(root/'inputs.json').write_text(json.dumps(dict(inputs=records,reference_sha256=sha(ref/'result.json'),runner_sha256=sha(__file__)),indent=2))
run([MS/'vlib.exe','work'],root,'vlib.log')
body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log')
assert 'Errors: 0, Warnings: 0' in body
body,seconds=run([MS/'vsim.exe','-c','work.tb_specialized_rejection','-do','onerror {quit -code 1}; run -all; quit -code 0'],root,'simulation.log',timeout=180)
lines=re.findall(r'^# (PASS SPECIALIZED_REJECTION cases=8 escaped_requests=0)$',body,re.M)
assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
for row in records:assert sha(row['snapshot'])==row['sha256']
result=dict(status='SPECIALIZED_REJECT_BEFORE_ACCESS_FOUR_STATE_PASS',line=lines[0],seconds=seconds,
 inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),full_frame=False)
(root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
