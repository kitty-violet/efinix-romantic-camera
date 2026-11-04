"""Fixed matrix production adapter, real resident image and unchanged tensor oracle."""
import argparse,collections,json,os,re,shutil,subprocess,sys,traceback
from pathlib import Path
from run_units import ROOT,HERE,MS,sha,run as sim_run
from fixed_variant_overlay import resolve_overlay
from simulation_primitives import with_public_lut4
sys.path.insert(0,str(HERE.parent/'pw_array_feed_abc'))
from run_native import BIN,VROOT,run
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--engine',choices=('compile','native','four-state'),default='native');p.add_argument('--simulator',choices=('modelsim','questa'),default='modelsim');p.add_argument('--dsp24-proof',type=Path);p.add_argument('--dispatch',type=Path);p.add_argument('--overrides-dir',type=Path);p.add_argument('--testbench',type=Path);p.add_argument('--frozen-board',type=Path);a=p.parse_args()
if a.simulator=='questa':
 MS=Path('E:/quatra/questa_fse/win64');assert (MS/'vsim.exe').is_file()
 os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
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
  if name=='compute_column_cluster.sv':f=HERE/'variants/fixed_production_cluster/row_pair_compute_cluster.sv'
  if name=='unified_column_dispatch32.sv':f=a.dispatch.resolve() if a.dispatch else HERE/'variants/ternary_stride/unified_column_dispatch32.sv'
  if name=='tb_compute_column_cluster.sv':f=HERE/'variants/fixed_production_cluster'/name
  sources.append(f)
 sources.extend(v/n for n in ('row_pair_column_reader.sv','row_pair_column_writer.sv','row_pair_skip_fuse4.sv','row_pair_aux_router.sv','row_pair_arena6.sv'))
 sources.extend(HERE/'variants/fixed_window_dot8x32'/n for n in (
  'fixed_video_compute.sv','fixed_production_mac.sv','fixed_source_columns6.sv','fixed_window_dot8x32.sv','fixed_ring_weights9.sv',
  'fixed_partial32.sv','fixed_rq32_mixed.sv','fixed_skip_vectors4.sv','fixed_column_store4.sv','fixed_weight_loader.sv'))
 sources=[HERE/'variants/row_pair_taps_fixed/row_pair_aux_router.sv' if f.name=='row_pair_aux_router.sv' else f for f in sources]
 if a.overrides_dir:
  overrides=resolve_overlay(a.overrides_dir)
  pipeline={'cnn_frame_datapath.sv','cnn_production_frame.sv','row_pair_dma.sv','frame_program_fetch.sv','frame_program_issue.sv'}
  assert not a.dispatch and overrides and set(overrides)<=set(f.name for f in sources)|pipeline,'Unknown, empty or conflicting overrides'
  sources=[overrides.get(f.name,f) for f in sources]
  if 'cnn_frame_datapath.sv' in overrides:
   # Arithmetic gate only: these pipeline changes require production tests.
   adapter=next(f for f in sources if f.name=='row_pair_compute_cluster.sv')
   test_folder='fixed_adapter_boundary_test' if 'AUTO_DW_GROUPS' in adapter.read_text() else 'fixed_next_weight_test'
   sources=[HERE/'variants'/test_folder/f.name if f.name=='tb_compute_column_cluster.sv' else f for f in sources]
 if a.testbench:
  tb=a.testbench.resolve();assert tb.name=='tb_compute_column_cluster.sv' and tb.is_relative_to(HERE/'variants')
  sources=[tb if f.name==tb.name else f for f in sources]
 board_binding=None
 if a.frozen_board:
  board=a.frozen_board.resolve();assert board.is_relative_to(ROOT/'tmp') and not a.overrides_dir and not a.dispatch
  bm=json.loads((board/'inputs.json').read_text())
  for rec in bm['inputs']:assert sha(rec['snapshot'])==rec['sha256']
  sources=[board/'assembly'/f.name if (board/'assembly'/f.name).is_file() else f for f in sources]
  board_binding=dict(path=str(board),inputs_sha256=sha(board/'inputs.json'),files={f.name:sha(f) for f in sources if f.is_relative_to(board)})
 for f in with_public_lut4(sources):
  dst=r/f.name;assert not dst.exists(),dst
  if primitive_proof and f.name=='efx_dsp24.v':
   assert sha(f)==primitive_proof['original_sha256'];f=pr/'efx_dsp24_flat.v'
  shutil.copy2(f,dst);records.append(dict(source=str(f),snapshot=str(dst),sha256=sha(f)))
 vectors=r/'vectors';shutil.copytree(ref/'vectors',vectors)
 manifest=dict(inputs=records,vectors={f.name:sha(f) for f in vectors.iterdir() if f.is_file()},golden_reference=str(ref),reference_inputs_sha256=sha(ref/'inputs.json'),runner_sha256=sha(__file__),engine=a.engine,simulator=a.simulator,dsp24_proof=primitive_proof,frozen_board=board_binding)
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
 result=dict(status='FIXED_CLUSTER_REAL_RESIDENT_PASS',engine=a.engine,dsp24_proof=primitive_proof,line=lines[0],seconds=sec,inputs_sha256=sha(r/'inputs.json'),log_sha256=sha(r/'simulation.log'),full_frame=False,whole_board=False)
 (r/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
except Exception as e:
 (r/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2)+'\n');raise
