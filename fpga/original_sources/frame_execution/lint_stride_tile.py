"""Structural loop/latch/range gate for the integrated stride consumer."""
import argparse,collections,json,os,re,sys,shutil
from pathlib import Path
from run_units import HERE,ROOT,EF,sha
from validate_stride_tile import selected
sys.path.insert(0,str(HERE.parent/'pw_array_feed_abc'))
from run_native import BIN,VROOT,run
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
paths=[EF/'sim_models/verilog'/n for n in ('efx_dsp24.v','efx_ram10.v','efx_dpram10.v')]+selected();records=[]
for f in paths:
 dst=root/f.name;shutil.copy2(f,dst);records.append(dict(path=str(f),snapshot=str(dst),sha256=sha(f)))
(root/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2))
env=os.environ.copy();env.update(PATH=str(BIN)+';'+env['PATH'],VERILATOR_ROOT=str(VROOT),TEMP=str(root),TMP=str(root))
body,seconds=run([BIN/'verilator_bin.exe','--lint-only','--timing','-Wno-fatal','--top-module','stride_tile_cluster','--timescale-override','1ns/1ps',*[r['snapshot'] for r in records]],root,root/'lint.log',env,timeout=180)
codes=collections.Counter(re.findall(r'^%Warning(?:-([A-Z0-9_]+))?:',body,re.M))
bad=set(codes)&{'UNOPTFLAT','SELRANGE','LATCH','IMPLICIT','PINMISSING'}
for r in records:assert sha(r['path'])==sha(r['snapshot'])==r['sha256']
result=dict(status='FAIL' if bad else 'STRIDE_TILE_STRUCTURAL_LINT_PASS_WIDTH_REVIEW_REQUIRED',warning_codes=dict(codes),bad=sorted(bad),inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'lint.log'),seconds=seconds,full_frame=False)
(root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));assert not bad and not re.search(r'^%Error',body,re.M)
