"""Static finite residual row router; no vendor-black-box or width suppression."""
import argparse,collections,json,os,re,sys,shutil
from pathlib import Path
from run_units import HERE,ROOT,sha
sys.path.insert(0,str(HERE.parent/'pw_array_feed_abc'))
from run_native import BIN,VROOT,run
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--row-source',type=Path,default=HERE/'residual_row_shared.sv');a=p.parse_args()
build=a.root.resolve();assert build.is_relative_to(ROOT/'tmp');build.mkdir(parents=True,exist_ok=False)
paths=[HERE/'residual_packet_dispatch32.sv',a.row_source.resolve()]
before={str(f):sha(f) for f in paths}
for f in paths+[Path(__file__)]:shutil.copy2(f,build/f.name)
env=os.environ.copy();env.update(PATH=str(BIN)+';'+env['PATH'],VERILATOR_ROOT=str(VROOT),TEMP=str(build),TMP=str(build))
body,seconds=run([BIN/'verilator_bin.exe','--lint-only','--timing','-Wno-fatal','--top-module','residual_row_shared',
 '--timescale-override','1ns/1ps',*[str(build/f.name) for f in paths]],build,build/'lint.log',env,timeout=180)
codes=collections.Counter(re.findall(r'^%Warning(?:-([A-Z0-9_]+))?:',body,re.M))
bad=set(codes)&{'UNOPTFLAT','SELRANGE','LATCH','IMPLICIT','PINMISSING','MULTIDRIVEN'}
for f,h in before.items():assert sha(f)==h
result=dict(status='FAIL' if bad else 'LOCAL_ROW_LINT_ZERO_WARNING_PASS' if not codes else 'LOCAL_ROW_LINT_STRUCTURAL_PASS_WIDTH_REVIEW_OPEN',warning_codes=dict(codes),bad=sorted(bad),
 source_hashes=before,log_sha256=sha(build/'lint.log'),seconds=seconds,full_frame=False)
(build/'result.json').write_text(json.dumps(result,indent=2));print(result)
assert not bad and not re.search(r'^%Error',body,re.M)

