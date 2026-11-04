"""Official dual-port RAM, independent pixel oracle, resets and frame markers."""
import argparse,json,re,shutil,os,sys,collections
from pathlib import Path
from run_units import ROOT,HERE,EF,MS,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--reference',action='store_true');p.add_argument('--full',action='store_true');p.add_argument('--native',action='store_true');p.add_argument('--source',type=Path);a=p.parse_args()
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
v=HERE/'variants/raw_row_upscale';base=ROOT/'tmp/frame-execution/connected-board-packed-display1/rtl'
srcs=[(base if a.reference else v)/'rgb_bilinear_scaler_prototype.sv',v/'tb_raw_row_upscale.sv',base/'resize_axis_up_2_to_3.sv',base/'resize_axis_down_3_to_2.sv',EF/'sim_models/verilog/efx_dpram10.v']
if a.source:
 assert not a.reference and a.source.name=='rgb_bilinear_scaler_prototype.sv' and a.source.resolve().is_relative_to(HERE/'variants')
 srcs[0]=a.source.resolve()
records=[]
for s in srcs:
 d=r/s.name;shutil.copy2(s,d);records.append(dict(source=str(s),snapshot=str(d),sha256=sha(s)))
(r/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__),reference=a.reference,full=a.full,native=a.native),indent=2))
run([MS/'vlib.exe','work'],r,'vlib.log',30)
body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[x['snapshot'] for x in records]],r,'compile.log',30)
assert 'Errors: 0, Warnings: 0' in body
if a.native:
 sys.path.insert(0,str(HERE.parent/'pw_array_feed_abc'))
 from run_native import BIN,VROOT,run as native_run
 env=os.environ.copy();env.update(PATH=str(BIN)+';E:\\GIT\\Git\\usr\\bin;'+env['PATH'],TEMP=str(r),TMP=str(r),TMPDIR=str(r),MSYS2_ARG_CONV_EXCL='*',VERILATOR_ROOT=str(VROOT),MAKE=str(BIN/'mingw32-make.exe'))
 obj=r/'obj';top='tb_raw_row_upscale'
 args=[BIN/'verilator_bin.exe','--cc','--exe','--main','--timing','--sched-zero-delay','--top-module',top,'--timescale-override','1ns/1ps','-Wno-fatal','--Mdir',obj.as_posix()]
 if a.full:args+=['-GSW=1280','-GSH=720']
 body,_=native_run(args+[Path(x['snapshot']).as_posix() for x in records],r,r/'native_compile.log',env)
 warnings=collections.Counter(re.findall(r'^%Warning(?:-([A-Z0-9_]+))?:',body));assert not(set(warnings)&{'UNOPTFLAT','LATCH','SELRANGE','IMPLICIT'}),warnings
 native_run([BIN/'mingw32-make.exe','-f','V'+top+'.mk','-j4',f'CXX={(BIN/"g++.exe").as_posix()}',f'LINK={(BIN/"g++.exe").as_posix()}',f'AR={(BIN/"ar.exe").as_posix()}','PERL=E:/GIT/Git/usr/bin/perl.exe',f'PYTHON3={Path(sys.executable).as_posix()}','SHELL=E:/GIT/Git/usr/bin/sh.exe',f'VERILATOR_ROOT={VROOT.as_posix()}','OPT_FAST=-O2','OPT_GLOBAL=-O2','CXXFLAGS=-std=gnu++20 -Werror=invalid-pch'],obj,r/'build.log',env)
 body,secs=native_run([obj/('V'+top+'.exe')],r,r/'simulation.log',env,timeout=240)
else:
 args=[MS/'vsim.exe','-c','work.tb_raw_row_upscale']
 if a.full:args+=['-gSW=1280','-gSH=720']
 body,secs=run(args+['-do','onerror {quit -code 1}; run -all; quit -code 0'],r,'simulation.log',240 if a.full else 90)
 assert 'Errors: 0, Warnings: 0' in body
lines=re.findall(r'^(?:# )?(PASS RAW_ROW_UPSCALE.*)$',body,re.M)
assert len(lines)==2
for x in records:assert sha(x['source'])==sha(x['snapshot'])==x['sha256']
result=dict(status='RAW_ROW_UPSCALE_NATIVE_PIXEL_PASS' if a.native else 'RAW_ROW_UPSCALE_OFFICIAL_PIXEL_PASS',lines=lines,seconds=secs,reference=a.reference,full=a.full,inputs_sha256=sha(r/'inputs.json'),log_sha256=sha(r/'simulation.log'),board=False)
(r/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
