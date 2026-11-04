"""Native actual-geometry 3-frame acquisition regression, after four-state gates.
This measures camera ingress -> RGB DDR commit, not the CNN/display interval.
"""
import argparse,collections,json,os,re,shutil,sys,traceback
from pathlib import Path
from run_units import ROOT,HERE,sha
sys.path.insert(0,str(HERE.parent/'pw_array_feed_abc'))
from run_native import BIN,VROOT,run
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--proof',type=Path,required=True);p.add_argument('--prefix-ns',type=int,default=0);p.add_argument('--timeout',type=float,default=180);a=p.parse_args()
r=a.root.resolve();proof=a.proof.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists()
gate=json.loads((proof/'result.json').read_text());assert gate['status']=='RAW_STREAM_FRONTEND_PASS' and len(gate['results'])==3 and sha(proof/'inputs.json')==gate['inputs_sha256']
r.mkdir();rows=[]
for row in json.loads((proof/'inputs.json').read_text())['inputs']:
 f=Path(row['snapshot']);assert sha(f)==row['sha256']
 if f.name=='tb_raw_stream_frontend.sv':f=HERE/'variants/raw_stream_test'/f.name
 d=r/f.name;shutil.copy2(f,d);rows.append(dict(source=str(f),snapshot=str(d),sha256=sha(d)))
(r/'inputs.json').write_text(json.dumps(dict(inputs=rows,runner_sha256=sha(__file__),four_state_gate_sha256=sha(proof/'result.json'),geometry=[1920,1080,1280,720],frames=3),indent=2)+'\n')
try:
 env=os.environ.copy();env.update(PATH=str(BIN)+';E:\\GIT\\Git\\usr\\bin;'+env['PATH'],TEMP=str(r),TMP=str(r),TMPDIR=str(r),MSYS2_ARG_CONV_EXCL='*',VERILATOR_ROOT=str(VROOT),MAKE=str(BIN/'mingw32-make.exe'))
 obj=r/'obj';top='tb_raw_stream_frontend'
 b,_=run([BIN/'verilator_bin.exe','--cc','--exe','--main','--timing','--sched-zero-delay','--top-module',top,'-GFULL=1','--timescale-override','1ns/1ps','--output-split','15000','--output-split-cfuncs','500','-Wno-fatal','--Mdir',obj.as_posix(),*[Path(x['snapshot']).as_posix() for x in rows]],r,r/'compile.log',env)
 warnings=collections.Counter(re.findall(r'^%Warning(?:-([A-Z0-9_]+))?:',b));assert not(set(warnings)&{'UNOPTFLAT','LATCH','SELRANGE','IMPLICIT'}),warnings
 run([BIN/'mingw32-make.exe','-f','V'+top+'.mk','-j4',f'CXX={(BIN/"g++.exe").as_posix()}',f'LINK={(BIN/"g++.exe").as_posix()}',f'AR={(BIN/"ar.exe").as_posix()}','PERL=E:/GIT/Git/usr/bin/perl.exe',f'PYTHON3={Path(sys.executable).as_posix()}','SHELL=E:/GIT/Git/usr/bin/sh.exe',f'VERILATOR_ROOT={VROOT.as_posix()}','OPT_FAST=-O2','OPT_GLOBAL=-O2','CXXFLAGS=-std=gnu++20 -Werror=invalid-pch'],obj,r/'build.log',env)
 binary=obj/('V'+top+'.exe')
 b,t=run([binary,*([f'+PREFIX_NS={a.prefix_ns}'] if a.prefix_ns else [])],r,r/'simulation.log',env,timeout=a.timeout)
 if a.prefix_ns:
  assert re.search(r'^PREFIX_ONLY ',b,re.M) and not re.search(r'FAIL|%Error',b)
  result=dict(status='RAW_STREAM_ACTIVE_PREFIX_ONLY',seconds=t,prefix_ns=a.prefix_ns,inputs_sha256=sha(r/'inputs.json'),log_sha256=sha(r/'simulation.log'),full_frame_pass=False)
  (r/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));sys.exit(0)
 lines=re.findall(r'^(PASS RAW_STREAM_FRONTEND.*)$',b,re.M);assert len(lines)==1 and not re.search(r'FAIL|%Error',b)
 frames=[dict(zip(['frame_id','input_start_ns','input_end_ns','rgb_ddr_commit_ns'],map(int,m))) for m in re.findall(r'FRONTEND_FRAME id=(\d+) input_start_ns=(\d+) input_end_ns=(\d+) rgb_ddr_commit_ns=(\d+)',b)]
 assert len(frames)==3
 for row in rows:assert sha(row['source'])==sha(row['snapshot'])==row['sha256']
 result=dict(status='RAW_STREAM_FULL_GEOMETRY_THREE_FRAME_PASS',line=lines[0],seconds=t,frames=frames,inputs_sha256=sha(r/'inputs.json'),log_sha256=sha(r/'simulation.log'),binary_sha256=sha(binary),four_state=False,ddr='behavioral response model, no other masters',cnn_included=False,display_included=False,physical_timing_pass=False)
 (r/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
except Exception as e:
 (r/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
