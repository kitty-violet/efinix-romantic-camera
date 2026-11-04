"""Apply the cycle-equivalent shared scaler to a frozen whole-board candidate."""
import argparse,json,shutil
from pathlib import Path
from run_units import ROOT,HERE,sha
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
base=a.base.resolve();root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
gate=ROOT/'tmp/frame-execution/shared-fixed-scaler1';proof=json.loads((gate/'result.json').read_text())
assert proof['status']=='FIXED_SCALER_EXHAUSTIVE_AND_CYCLE_AB_PASS'
assert sha(gate/'inputs.json')==proof['inputs_sha256'] and sha(gate/'simulation.log')==proof['log_sha256']
pi=json.loads((gate/'inputs.json').read_text())
for x in pi['inputs']:assert sha(x['snapshot'])==x['sha256']
candidate=HERE/'variants/shared_fixed_scaler/rgb_bilinear_scaler_prototype.sv'
assert sha(candidate)==sha(gate/candidate.name)
original=(gate/'reference.sv').read_text().replace('module rgb_bilinear_scaler_reference','module rgb_bilinear_scaler_prototype',1)
assert original==(base/'rtl'/candidate.name).read_text()
meta=json.loads((base/'inputs.json').read_text());prior=json.loads((base/'result.json').read_text())
assert sha(base/'inputs.json')==prior['inputs_sha256']
root.mkdir();records=[];changed=0
for folder in ('rtl','assembly'):shutil.copytree(base/folder,root/folder)
for row in meta['inputs']:
 f=Path(row['snapshot']);assert sha(f)==row['sha256'];dst=root/f.relative_to(base);dst.parent.mkdir(parents=True,exist_ok=True)
 source=candidate if f.name==candidate.name else f
 shutil.copy2(source,dst);changed+=int(f.name==candidate.name)
 records.append(dict(source=str(source),snapshot=str(dst),sha256=sha(dst),base_sha256=row['sha256']))
assert changed==1
meta.update(inputs=records,base=str(base),base_inputs_sha256=sha(base/'inputs.json'),
 source_files=[str(root/Path(f).relative_to(base)) for f in meta['source_files']],
 shared_scaler=dict(proof=str(gate),result_sha256=sha(gate/'result.json'),source_sha256=sha(candidate)),
 runner_sha256=sha(__file__),board_release=False)
(root/'inputs.json').write_text(json.dumps(meta,indent=2)+'\n')
assert sum(x['sha256']!=x['base_sha256'] for x in records)==1
print('SHARED_SCALER_BOARD_FROZEN',len(records),flush=True)
