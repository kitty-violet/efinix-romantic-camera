"""Reuse a frozen, compiled full-geometry frontend after an insufficient host budget."""
import argparse,json,re,sys,traceback
from pathlib import Path
from run_units import ROOT,sha
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'pw_array_feed_abc'))
from run_native import run,BIN
import os
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--timeout',type=int,default=900);a=p.parse_args()
base=a.base.resolve();root=a.root.resolve()
assert base.is_relative_to(ROOT/'tmp') and root.is_relative_to(ROOT/'tmp') and not root.exists()
failure=json.loads((base/'failure.json').read_text());assert 'TimeoutExpired' in failure['error']
meta=json.loads((base/'inputs.json').read_text());assert meta['geometry']==[1920,1080,1280,720] and meta['frames']==3
for row in meta['inputs']:assert sha(row['snapshot'])==row['sha256']
exe=base/'obj/Vtb_raw_stream_frontend.exe';assert exe.is_file() and 180<a.timeout<=900
root.mkdir();record=dict(base=str(base),base_inputs_sha256=sha(base/'inputs.json'),inputs=meta['inputs'],binary=str(exe),binary_sha256=sha(exe),timeout_seconds=a.timeout,runner_sha256=sha(__file__))
(root/'inputs.json').write_text(json.dumps(record,indent=2))
env=os.environ.copy();env.update(PATH=str(BIN)+';'+env['PATH'],TEMP=str(root),TMP=str(root),TMPDIR=str(root))
try:
 # The compiled binary needs its existing MinGW runtime DLL directory.
 # Check executable startup before spending the full-frame host budget.
 prefix,_=run([exe,'+PREFIX_NS=10000'],root,root/'startup.log',env,timeout=15)
 assert re.search(r'^PREFIX_ONLY ',prefix,re.M) and not re.search(r'FAIL|%Error',prefix)
 body,secs=run([exe],root,root/'simulation.log',env,timeout=a.timeout)
 lines=re.findall(r'^(PASS RAW_STREAM_FRONTEND.*)$',body,re.M);assert len(lines)==1 and not re.search(r'FAIL|%Error',body)
 frames=[dict(zip(['frame_id','input_start_ns','input_end_ns','rgb_ddr_commit_ns'],map(int,m))) for m in re.findall(r'FRONTEND_FRAME id=(\d+) input_start_ns=(\d+) input_end_ns=(\d+) rgb_ddr_commit_ns=(\d+)',body)]
 assert len(frames)==3 and sha(exe)==record['binary_sha256']
 for row in meta['inputs']:assert sha(row['snapshot'])==row['sha256']
 result=dict(status='RAW_STREAM_FULL_GEOMETRY_THREE_FRAME_PASS',line=lines[0],frames=frames,seconds=secs,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),binary_sha256=sha(exe),four_state=False,ddr='behavioral response model, no other masters',cnn_included=False,display_included=False,physical_timing_pass=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
except Exception as e:
 (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
