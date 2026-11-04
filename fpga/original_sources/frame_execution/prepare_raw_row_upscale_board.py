"""Bind the two-raw-row upscaler's four-state and full-size pixel proofs to a board."""
import argparse,json,shutil
from pathlib import Path
from run_units import ROOT,HERE,sha
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
base=a.base.resolve();root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
name='rgb_bilinear_scaler_prototype.sv'
candidate=HERE/'variants/raw_row_upscale'/name
bindings=[]
for folder,status,full,reference in (
 ('raw-row-upscale2','RAW_ROW_UPSCALE_OFFICIAL_PIXEL_PASS',False,False),
 ('raw-row-upscale-full1','RAW_ROW_UPSCALE_NATIVE_PIXEL_PASS',True,False),
 ('raw-row-upscale-reference-full1','RAW_ROW_UPSCALE_NATIVE_PIXEL_PASS',True,True)):
 gate=ROOT/'tmp/frame-execution'/folder;gr=json.loads((gate/'result.json').read_text());gm=json.loads((gate/'inputs.json').read_text())
 assert gr['status']==status and gr['full']==full and gr['reference']==reference
 assert sha(gate/'inputs.json')==gr['inputs_sha256'] and sha(gate/'simulation.log')==gr['log_sha256']
 for row in gm['inputs']:assert sha(row['snapshot'])==row['sha256']
 assert len(gr['lines'])==2 and 'BAD_MARKER_RESET_RECOVERY' in gr['lines'][1]
 if full:assert 'frames=3 pixels=6220800 width=1920 height=1080' in gr['lines'][0]
 assert sha(gate/name)==sha(base/'rtl'/name if reference else candidate)
 assert sha(gate/'resize_axis_up_2_to_3.sv')==sha(base/'rtl/resize_axis_up_2_to_3.sv')
 bindings.append(dict(path=str(gate),result_sha256=sha(gate/'result.json')))
meta=json.loads((base/'inputs.json').read_text());prior=json.loads((base/'result.json').read_text())
assert sha(base/'inputs.json')==prior['inputs_sha256']
root.mkdir();records=[]
for folder in ('rtl','assembly'):shutil.copytree(base/folder,root/folder)
for row in meta['inputs']:
 f=Path(row['snapshot']);assert sha(f)==row['sha256'];dst=root/f.relative_to(base);dst.parent.mkdir(parents=True,exist_ok=True)
 source=candidate if f.name==name else f;shutil.copy2(source,dst)
 records.append(dict(source=str(source),snapshot=str(dst),sha256=sha(dst),base_sha256=row['sha256']))
assert sum(x['sha256']!=x['base_sha256'] for x in records)==1
meta.update(inputs=records,base=str(base),base_inputs_sha256=sha(base/'inputs.json'),
 source_files=[str(root/Path(f).relative_to(base)) for f in meta['source_files']],
 raw_row_upscale=dict(proofs=bindings,source_sha256=sha(candidate),scope='Fixed 2:3 separable bilinear with unchanged two-stage rounding; cycle count may change'),
 runner_sha256=sha(__file__),board_release=False)
(root/'inputs.json').write_text(json.dumps(meta,indent=2)+'\n')
print('RAW_ROW_UPSCALE_BOARD_FROZEN',len(records),flush=True)
