"""Freeze a whole-board synthesis-option A/B; no RTL, clock or pin changes."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,sha
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--mode',choices=['speed','area','area2','cong']);p.add_argument('--ce-effort',type=int,choices=range(5));p.add_argument('--min-ce-fanout',type=int);p.add_argument('--retiming',type=int,choices=[0,1,2]);a=p.parse_args()
base=a.base.resolve();root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
meta=json.loads((base/'inputs.json').read_text());result=json.loads((base/'result.json').read_text())
assert sha(base/'inputs.json')==result['inputs_sha256']
for r in meta['inputs']:assert sha(r['snapshot'])==r['sha256']
root.mkdir();records=[]
for d in ('rtl','assembly'):shutil.copytree(base/d,root/d)
for f in base.iterdir():
 if f.is_file() and f.suffix in ('.xml','.sdc','.bin','.v','.sv','.vh','.svh'):shutil.copy2(f,root/f.name)
for r in meta['inputs']:
 old=Path(r['snapshot']);dst=root/old.relative_to(base);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old,dst)
 records.append(dict(source=str(old),snapshot=str(dst),sha256=sha(dst),base_sha256=r['sha256']))
xml=root/'ti60f225_oob.xml';body=xml.read_text();changes=[]
for name,val in [('mode',a.mode),('infer-clk-enable',a.ce_effort),('min-ce-fanout',a.min_ce_fanout),('retiming',a.retiming)]:
 if val is None:continue
 if name=='min-ce-fanout':assert val>=0
 pattern=r'(name="'+name+r'" value=")([^"]+)';matches=list(re.finditer(pattern,body));assert len(matches)==1,name
 changes.append(dict(name='synthesis.'+name,before=matches[0][2],after=str(val)))
 body=re.sub(pattern,lambda m:m[1]+str(val),body)
assert changes;xml.write_text(body.rstrip('\r\n'))
for r in records:
 if Path(r['snapshot'])==xml:r.update(source=str(xml),sha256=sha(xml))
meta.update(inputs=records,base=str(base),base_inputs_sha256=sha(base/'inputs.json'),source_files=[str(root/Path(f).relative_to(base)) for f in meta['source_files']],overrides=[],override=None,configuration_changes=changes,timing_constraints_changed=False,board_release=False)
(root/'inputs.json').write_text(json.dumps(meta,indent=2));print(json.dumps(dict(status='SYNTHESIS_TRIAL_FROZEN',changes=changes,inputs=len(records),inputs_sha256=sha(root/'inputs.json'))))
