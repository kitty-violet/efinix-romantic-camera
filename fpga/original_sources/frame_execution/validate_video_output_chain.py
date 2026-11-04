"""Small direct replay of the exact scaler/FIFO/scheduler with pixel checkers."""
import argparse,json,os,re,shutil,sys,traceback
from pathlib import Path
from run_units import ROOT,HERE,sha
sys.path.insert(0,str(HERE.parent/'pw_array_feed_abc'))
from run_native import BIN,VROOT,run
p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,required=True)
p.add_argument('--map',type=Path,required=True)
p.add_argument('--vectors',type=Path,required=True)
p.add_argument('--fixtures',type=Path,required=True)
p.add_argument('--ddr',action='store_true')
p.add_argument('--read-burst',type=int,default=16)
p.add_argument('--address-delay',type=int,default=0)
a=p.parse_args();root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp')
root.mkdir(parents=True,exist_ok=False)
try:
 m=json.loads((a.map/'inputs.json').read_text());available={Path(r['snapshot']).name:r for r in m['inputs']}
 paths=[]
 for name in ('rgb_bilinear_scaler_prototype.sv','resize_axis_up_2_to_3.sv','resize_axis_down_3_to_2.sv',
              'framed_rgb_async_fifo.sv','guarded_async_fifo.sv','DC_FIFO.v','cdc_mailbox.sv','hdmi_frame_scheduler.sv'):
  r=available[name];assert sha(r['snapshot'])==r['sha256'];paths.append(Path(r['snapshot']))
 paths += [HERE/n for n in ('rgb720_ddr_reader.sv','rgb24_stream_unpack.sv','tb_video_output_chain.sv')]
 records=[]
 for src in paths:
  dst=root/src.name;shutil.copy2(src,dst);records.append(dict(snapshot=str(dst),sha256=sha(dst)))
 (root/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__),
  ddr=a.ddr,read_burst=a.read_burst,address_delay=a.address_delay),indent=2))
 env=os.environ.copy();env.update(PATH=str(BIN)+';E:/GIT/Git/usr/bin;'+env['PATH'],TEMP=str(root),TMP=str(root),
  TMPDIR=str(root),MSYS2_ARG_CONV_EXCL='*',VERILATOR_ROOT=str(VROOT),MAKE=str(BIN/'mingw32-make.exe'))
 obj=root/'obj';top='tb_video_output_chain'
 body,_=run([BIN/'verilator_bin.exe','--cc','--exe','--main','--timing','--top-module',top,
  '-GDDR='+str(int(a.ddr)),'-GREAD_BURST='+str(a.read_burst),'-GADDRESS_DELAY='+str(a.address_delay),
  '--timescale-override','1ns/1ps','-Wno-fatal','--Mdir',obj.as_posix(),*[Path(r['snapshot']).as_posix() for r in records]],root,root/'compile.log',env)
 warnings=re.findall(r'^%Warning-([A-Z0-9_]+)',body,re.M)
 assert not(set(warnings)&{'UNOPTFLAT','LATCH','SELRANGE','IMPLICIT'}),warnings
 run([BIN/'mingw32-make.exe','-f','V'+top+'.mk','-j4',f'CXX={(BIN/"g++.exe").as_posix()}',
  f'LINK={(BIN/"g++.exe").as_posix()}',f'AR={(BIN/"ar.exe").as_posix()}','PERL=E:/GIT/Git/usr/bin/perl.exe',
  f'PYTHON3={Path(sys.executable).as_posix()}','SHELL=E:/GIT/Git/usr/bin/sh.exe',f'VERILATOR_ROOT={VROOT.as_posix()}',
  'OPT_FAST=-O2','OPT_GLOBAL=-O2','CXXFLAGS=-std=gnu++20 -Werror=invalid-pch'],obj,root/'build.log',env)
 vm=json.loads((a.vectors/'frame_manifest.json').read_text())
 offset=next(n['offset'] for n in vm['nodes'] if n['id']=='rgb_head')
 body,seconds=run([obj/('V'+top+'.exe'),'+VECTORS='+a.vectors.resolve().as_posix(),
  '+FIXTURES='+a.fixtures.resolve().as_posix(),'+FINAL_OFFSET='+str(offset)],root,root/'simulation.log',env,timeout=240)
 assert 'PASS VIDEO_OUTPUT_CHAIN' in body and not re.search(r'FAIL|%Error',body)
 result=dict(status='VIDEO_OUTPUT_CHAIN_NATIVE_PASS',seconds=seconds,inputs_sha256=sha(root/'inputs.json'),
  log_sha256=sha(root/'simulation.log'),full_system=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
except Exception as e:
 (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
