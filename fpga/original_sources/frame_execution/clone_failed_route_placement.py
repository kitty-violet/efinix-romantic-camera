"""Preserve a failed 70-iteration route and reuse its exact legal placement."""
import argparse,json,shutil
from pathlib import Path
from run_units import ROOT,sha

p=argparse.ArgumentParser();p.add_argument('--prefix',required=True,type=Path);p.add_argument('--root',required=True,type=Path)
a=p.parse_args();base=a.prefix.resolve();root=a.root.resolve()
assert base.is_relative_to(ROOT/'tmp/frame-execution') and root.is_relative_to(ROOT/'tmp/frame-execution') and not root.exists()
old=base/'place-seed1';text=(old/'outflow/ti60f225_oob.route.out').read_text()
assert 'Iter 70: overlap=1 ' in text and 'Routing FAILED; could not resolve congestion.' in text
assert not (old/'outflow/ti60f225_oob.troutingtraces').exists()
root.mkdir();new=root/'place-seed1';new.mkdir()
for folder in ('rtl','assembly','work_pnr'):
    shutil.copytree(old/folder,new/folder)
(new/'outflow').mkdir()
for f in (old/'outflow').iterdir():
    if f.is_file() and '.route' not in f.name:shutil.copy2(f,new/'outflow'/f.name)
for f in old.iterdir():
    if f.is_file() and (f.suffix in ('.xml','.sdc','.bin','.v','.sv','.vh','.svh','.ini','.placeloc')
                         or f.name in ('physical_runner.py','capacity_gate.json','placement_gate.json','dsp_placement.json')):
        shutil.copy2(f,new/f.name)
meta=json.loads((old/'physical_inputs.json').read_text())
for rec in meta['inputs']:
    rec['snapshot']=str(new/Path(rec['snapshot']).relative_to(old))
    assert sha(rec['snapshot'])==rec['sha256']
(new/'physical_inputs.json').write_text(json.dumps(meta,indent=2))
shutil.copy2(old/'before_route/result.json',new/'result.json')
shutil.copy2(base/'inputs.json',root/'inputs.json');shutil.copy2(base/'failure.json',root/'failure.json')
shutil.copytree(base/'startup',root/'startup')
preserved={}
for filename in ('ti60f225_oob.map.v','ti60f225_oob.place','ti60f225_oob.vdb','ti60f225_oob.out.placeloc'):
    assert sha(old/'outflow'/filename)==sha(new/'outflow'/filename)
    preserved[filename]=sha(new/'outflow'/filename)
result=dict(status='EXACT_PLACEMENT_REUSED_AFTER_70_ITERATION_FAILURE',parent=str(base),root=str(root),
            physical_files=preserved,failed_route_sha256=sha(old/'outflow/ti60f225_oob.route.out'),
            original_failure_preserved=True,hardware_access=False)
(root/'clone_receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
