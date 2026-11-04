"""Rebind an existing transport fixture after the async RAM mode correction."""
import argparse,json,shutil
from pathlib import Path
from run_units import ROOT,HERE,sha
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
src=a.source.resolve();dst=a.root.resolve();assert dst.is_relative_to(ROOT/'tmp') and not dst.exists()
old=json.loads((src/'inputs.json').read_text())
for f,h in old['files'].items():assert sha(f)==h,f
helper=HERE/'variants/video_service/video_dualclock_sample.sv'
assert (src/helper.name).read_text().replace('READ_FIRST','READ_UNKNOWN')==helper.read_text()
dst.mkdir()
for f in src.iterdir():
 if f.suffix in ('.v','.sv'):shutil.copy2(f,dst/f.name)
shutil.copy2(helper,dst/helper.name)
files={str(f):sha(f) for f in dst.iterdir()}
files.update({str(f):sha(f) for f in (src/'inputs.json',helper,Path(__file__))})
(dst/'inputs.json').write_text(json.dumps(dict(files=files,parent=str(src),parent_inputs_sha256=sha(src/'inputs.json'),change='Only async RAM WRITE_MODE READ_FIRST -> READ_UNKNOWN',hardware_access=False),indent=2))
print('RAM_FIXTURE_MODE_REBOUND',dst)
