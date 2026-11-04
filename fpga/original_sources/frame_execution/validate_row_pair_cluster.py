"""Real MAC/RQ/weights with row-pair SRAM; unchanged independent golden cases."""
import argparse,collections,json,os,re,shutil,subprocess,sys,traceback
from pathlib import Path
from run_units import ROOT,HERE,MS,sha,run as sim_run
sys.path.insert(0,str(HERE.parent/'pw_array_feed_abc'))
from run_native import BIN,VROOT,run
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--engine',choices=('compile','native','four-state'),default='native');p.add_argument('--dsp24-proof',type=Path);p.add_argument('--dispatch',type=Path);p.add_argument('--overrides-dir',type=Path);a=p.parse_args()
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
try:
 ref=ROOT/'tmp/frame-execution/header-ram-cluster2/frozen';src=json.loads((ref/'inputs.json').read_text())
 v=HERE/'variants/row_pair_arena';records=[];sources=[];primitive_proof=None
 if a.dsp24_proof:
  pr=a.dsp24_proof.resolve();proof=json.loads((pr/'result.json').read_text());pi=json.loads((pr/'inputs.json').read_text())
  assert proof['status']=='DSP24_BOARD_PIPE_HIERARCHY_EQUIVALENCE_PASS' and proof['compact_board']
  assert sha(pr/'inputs.json')==proof['inputs_sha256'] and sha(pr/'simulation.log')==proof['log_sha256']
  assert sha(pr/'efx_dsp24_flat.v')==proof['flat_sha256']==pi['flat_sha256']
  primitive_proof=dict(path=str(pr),result_sha256=sha(pr/'result.json'),original_sha256=pi['original_sha256'],view_sha256=proof['flat_sha256'])
 for rec in src['inputs']:
  f=Path(rec['snapshot']);assert sha(f)==rec['sha256'];name=f.name
  if name=='compute_column_cluster.sv':f=v/'row_pair_compute_cluster.sv'
  if name=='unified_column_dispatch32.sv':f=a.dispatch.resolve() if a.dispatch else HERE/'variants/ternary_stride/unified_column_dispatch32.sv'
  if name=='tb_compute_column_cluster.sv':f=v/name
  sources.append(f)
 sources.extend(v/n for n in ('row_pair_column_reader.sv','row_pair_column_writer.sv','row_pair_skip_fuse4.sv','row_pair_aux_router.sv','row_pair_arena6.sv'))
 if a.overrides_dir:
  overrides={f.name:f for f in a.overrides_dir.resolve().glob('*.sv')}
  assert not a.dispatch and overrides and set(overrides)<=set(f.name for f in sources),'Unknown, empty or conflicting overrides'
  sources=[overrides.get(f.name,f) for f in sources]
 for f in sources:
  dst=r/f.name;assert not dst.exists(),dst
  if primitive_proof and f.name=='efx_dsp24.v':
   assert sha(f)==primitive_proof['original_sha256'];f=pr/'efx_dsp24_flat.v'
  shutil.copy2(f,dst);records.append(dict(source=str(f),snapshot=str(dst),sha256=sha(f)))
 vectors=r/'vectors';shutil.copytree(ref/'vectors',vectors)
 manifest=dict(inputs=records,vectors={f.name:sha(f) for f in vectors.iterdir() if f.is_file()},golden_reference=str(ref),reference_inputs_sha256=sha(ref/'inputs.json'),runner_sha256=sha(__file__),engine=a.engine,dsp24_proof=primitive_proof)
 (r/'inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
 sim_run([MS/'vlib.exe','work'],r,'vlib.log')
 body,_=sim_run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[i['snapshot'] for i in records]],r,'fourstate_compile.log');assert 'Errors: 0, Warnings: 0' in body
 if a.engine=='compile':
  (r/'result.json').write_text(json.dumps(dict(status='ROW_PAIR_CLUSTER_COMPILE_ONLY',inputs_sha256=sha(r/'inputs.json'),numeric_pass=False),indent=2)+'\n');print('ROW_PAIR_CLUSTER_COMPILE_ONLY');sys.exit(0)
 if a.engine=='native':
  env=os.environ.copy();env.update(PATH=str(BIN)+';E:\\GIT\\Git\\usr\\bin;'+env['PATH'],TEMP=str(r),TMP=str(r),TMPDIR=str(r),MSYS2_ARG_CONV_EXCL='*',VERILATOR_ROOT=str(VROOT),MAKE=str(BIN/'mingw32-make.exe'))
  obj=r/'obj';top='tb_compute_column_cluster'
  body,_=run([BIN/'verilator_bin.exe','--cc','--exe','--main','--timing','--sched-zero-delay','--top-module',top,'--timescale-override','1ns/1ps','--output-split','15000','--output-split-cfuncs','500','-Wno-fatal','--Mdir',obj.as_posix(),*[Path(i['snapshot']).as_posix() for i in records]],r,r/'compile.log',env)
  warnings=collections.Counter(re.findall(r'^%Warning(?:-([A-Z0-9_]+))?:',body));assert not(set(warnings)&{'UNOPTFLAT','LATCH','SELRANGE','IMPLICIT'}),warnings
  run([BIN/'mingw32-make.exe','-f','V'+top+'.mk','-j4',f'CXX={(BIN/"g++.exe").as_posix()}',f'LINK={(BIN/"g++.exe").as_posix()}',f'AR={(BIN/"ar.exe").as_posix()}','PERL=E:/GIT/Git/usr/bin/perl.exe',f'PYTHON3={Path(sys.executable).as_posix()}','SHELL=E:/GIT/Git/usr/bin/sh.exe',f'VERILATOR_ROOT={VROOT.as_posix()}','OPT_FAST=-O2','OPT_GLOBAL=-O2','CXXFLAGS=-std=gnu++20 -Werror=invalid-pch'],obj,r/'build.log',env)
  binary=obj/('V'+top+'.exe');command=subprocess.list2cmdline([sys.executable,'-B',str(HERE.parent/'coefficient_store/validate_readback.py'),'--image-dir',str(vectors),'--readback',str(r/'readback.hex')])
  body,sec=run([binary,'+VECTORS='+vectors.as_posix(),'+DUMP='+(r/'readback.hex').as_posix(),'+VERIFY_COMMAND='+command],r,r/'simulation.log',env,timeout=180)
 else:
  do='onerror {quit -code 1}\nrun 120 us\n'
  do+='if {[examine -radix unsigned sim:/tb_compute_column_cluster/verify_checkpoint] != 1} {error "Missing actual readback"}\n'
  do+='puts [exec {'+Path(sys.executable).as_posix()+'} -B {'+(HERE.parent/'coefficient_store/validate_readback.py').as_posix()+'} --image-dir {'+vectors.as_posix()+'} --readback {'+(r/'readback.hex').as_posix()+'}]\n'
  do+='force -deposit sim:/tb_compute_column_cluster/software_verified 1\nrun -all\nquit -code 0\n';(r/'run.do').write_text(do)
  body,sec=sim_run([MS/'vsim.exe','-c','work.tb_compute_column_cluster','+VECTORS='+vectors.as_posix(),'+DUMP='+(r/'readback.hex').as_posix(),'-do','do run.do'],r,'simulation.log',timeout=240)
 lines=re.findall(r'^(?:# )?(PASS COMPUTE_CLUSTER.*)$',body,re.M);assert len(lines)==1 and 'ACTUAL_RTL_READBACK_SOFTWARE_VERIFIED' in body and not re.search(r'\bFAIL\b|%Error',body)
 for i in records:assert sha(i['source'])==sha(i['snapshot'])==i['sha256']
 result=dict(status='ROW_PAIR_ACTUAL_MAC_RQ_CLUSTER_PASS',engine=a.engine,dsp24_proof=primitive_proof,line=lines[0],seconds=sec,inputs_sha256=sha(r/'inputs.json'),log_sha256=sha(r/'simulation.log'),full_frame=False,whole_board=False)
 (r/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
except Exception as e:
 (r/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2)+'\n');raise
