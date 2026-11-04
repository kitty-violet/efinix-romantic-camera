"""Directed integrated datapath guard verification; no mock compute or DMA."""
import argparse,collections,json,os,re,shutil,sys,traceback
from pathlib import Path
from run_units import HERE,ROOT,EF,sha
from map_parallel_candidates import selection
sys.path.insert(0,str(HERE.parent/'pw_array_feed_abc'))
from run_native import BIN,VROOT,run
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--active',action='store_true');p.add_argument('--overrides-dir',type=Path);p.add_argument('--frozen-base',type=Path);p.add_argument('--dsp-model',action='store_true');a=p.parse_args()
root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
(root/'frozen_runner.py').write_bytes(Path(__file__).read_bytes())
try:
 paths=[EF/'sim_models/verilog'/n for n in ('efx_dsp24.v','efx_ram10.v','efx_dpram10.v')]+selection('datapath')[1]+[HERE/'tb_cnn_datapath_guard.sv']
 from frozen_sources import frozen_paths
 paths=frozen_paths(paths,a.frozen_base)
 if a.overrides_dir:
  override=a.overrides_dir.resolve();assert override.is_relative_to(HERE/'variants')
  paths=[override/f.name if (override/f.name).is_file() else f for f in paths]
 if a.dsp_model:paths.insert(0,EF/'sim_models/verilog/efx_dsp48.v')
 records=[]
 for f in paths:
  dst=root/f.name;shutil.copy2(f,dst);records.append(dict(path=str(f),snapshot=str(dst),sha256=sha(dst)))
 env=os.environ.copy();env.update(PATH=str(BIN)+';E:/GIT/Git/usr/bin;'+env['PATH'],TEMP=str(root),TMP=str(root),TMPDIR=str(root),MSYS2_ARG_CONV_EXCL='*',VERILATOR_ROOT=str(VROOT),MAKE=str(BIN/'mingw32-make.exe'))
 top='tb_cnn_datapath_guard';obj=root/'obj'
 args=[BIN/'verilator_bin.exe','--cc','--exe','--main','--timing','--top-module',top,'-GACTIVE_TEST='+str(int(a.active)),'--timescale-override','1ns/1ps','-Wno-fatal','--Mdir',obj.as_posix(),*[Path(r['snapshot']).as_posix() for r in records]]
 body,_=run(args,root,root/'compile.log',env);warnings=collections.Counter(re.findall(r'^%Warning(?:-([A-Z0-9_]+))?:',body,re.M))
 assert not(set(warnings)&{'UNOPTFLAT','LATCH','SELRANGE','IMPLICIT'}),warnings
 make=[BIN/'mingw32-make.exe','-f','V'+top+'.mk','-j4',f'CXX={(BIN/"g++.exe").as_posix()}',f'LINK={(BIN/"g++.exe").as_posix()}',f'AR={(BIN/"ar.exe").as_posix()}','PERL=E:/GIT/Git/usr/bin/perl.exe',f'PYTHON3={Path(sys.executable).as_posix()}','SHELL=E:/GIT/Git/usr/bin/sh.exe',f'VERILATOR_ROOT={VROOT.as_posix()}','OPT_FAST=-O1','OPT_GLOBAL=-O1','CXXFLAGS=-std=gnu++20 -Werror=invalid-pch']
 run(make,obj,root/'build.log',env);binary=obj/('V'+top+'.exe')
 manifest=dict(inputs=records,runner_sha256=sha(root/'frozen_runner.py'),binary_sha256=sha(binary),warning_codes=dict(warnings))
 (root/'inputs.json').write_text(json.dumps(manifest,indent=2))
 body,seconds=run([binary],root,root/'simulation.log',env,timeout=180)
 lines=re.findall(r'^PASS DATAPATH_GUARD.*$',body,re.M);assert len(lines)==1 and 'FAIL' not in body
 active_lines=re.findall(r'^PASS DATAPATH_ACTIVE_CANCEL.*$',body,re.M)
 assert len(active_lines)==int(a.active)
 for r in records:assert sha(r['path'])==sha(r['snapshot'])==r['sha256']
 result=dict(status='INTEGRATED_DATAPATH_GUARD_NATIVE_PASS',line=lines[0],active_lines=active_lines,seconds=seconds,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),four_state=False,production_scheduler=False,full_frame=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
except Exception as e:
 (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
