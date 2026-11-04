"""Directed integrated datapath guard verification; no mock compute or DMA."""
import argparse,collections,json,os,re,shutil,sys,traceback
from pathlib import Path
from run_units import HERE,ROOT,EF,sha
from map_parallel_candidates import selection
from fixed_variant_overlay import resolve_overlay
sys.path.insert(0,str(HERE.parent/'pw_array_feed_abc'))
from run_native import BIN,VROOT,run
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--active',action='store_true');p.add_argument('--address-fault',action='store_true');p.add_argument('--fault-boundary',action='store_true');p.add_argument('--overrides-dir',type=Path);p.add_argument('--frozen-base',type=Path);p.add_argument('--dsp-model',action='store_true');a=p.parse_args()
assert not a.address_fault or a.active,'Address fault gate must include active compute recovery'
root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
(root/'frozen_runner.py').write_bytes(Path(__file__).read_bytes())
try:
 fixed_window=os.environ.get('FPGA_FIXED_WINDOW_PRODUCTION')=='1'
 ref=Path(os.environ.get('FPGA_FIXED_REFERENCE',str(ROOT/'tmp/frame-execution/fixed-production-short1'))) if fixed_window else ROOT/'tmp/frame-execution/row-pair-production-smoke1'
 manifest=json.loads((ref/'inputs.json').read_text());paths=[]
 for rec in manifest['inputs']:
  f=Path(rec['snapshot']);assert sha(f)==rec['sha256']
  if f.name=='tb_fourway_full.sv':continue
  paths.append(f)
 paths.append(HERE/'variants/row_pair_arena/tb_cnn_datapath_guard.sv')
 assert a.frozen_base is None,'This row-pair runner uses its explicit validated frozen base'
 if a.overrides_dir:
  overrides=resolve_overlay(a.overrides_dir)
  assert overrides and set(overrides)<=set(f.name for f in paths),'Unknown or empty overrides'
  paths=[overrides.get(f.name,f) for f in paths]
 records=[]
 for f in paths:
  dst=root/f.name
  if fixed_window and f.name=='tb_cnn_datapath_guard.sv':
   body=f.read_text()
   aliases={'wrerror':'load_error','uerror':'engine.mac.error','fuse_error':'engine.skip_error','expand_error':'engine.writer_error','pool_busy':'engine.rq_busy'}
   for old,new in aliases.items():body=body.replace('dut.dut.'+old,'dut.dut.'+new)
   body=body.replace('active_stage=3;launch(0,32\'h00000100);wait(arvalid);','active_stage=3;launch(0,32\'h00000100);wait(arvalid);\n   wait(dut.dut.engine.mac.cv);')
   if any(f.name=='cnn_frame_datapath.sv' and 'NEXT_WEIGHT_PREFETCH' in f.read_text() for f in paths):
    body=body.replace('cnn_frame_datapath #(.WATCHDOG_CYCLES(ACTIVE_TEST ? 256:24)) dut(',
      "cnn_frame_datapath #(.WATCHDOG_CYCLES(ACTIVE_TEST ? 256:24),.NEXT_WEIGHT_PREFETCH(1)) dut(\n .next_weight_valid(1'b1),.next_weight_pw(1'b1),.next_weight_stem(1'b0),.next_weight_groups(3'd4),.next_weight_offset(15'd1792),")
    body=body.replace('wait(core_busy && dut.dut.engine.rq_busy && hbusy && dbusy && arvalid);',
      'wait(core_busy && dut.dut.engine.rq_busy && hbusy && dbusy && arvalid && dut.dut.preload_started && dut.dut.load_busy);')
    body=body.replace('simultaneous_compute_header_dma=1 restored_words=', 'simultaneous_compute_header_dma=1 next_weight_loader_active_at_cancel=1 restored_words=')
   if a.fault_boundary:
    marker='  $display("PASS DATAPATH_GUARD cases='
    assert body.count(marker)==1
    injection='''  // Target the global stop/commit contract independently of each origin's
  // error detection tests. Hold a real accepted DMA AR while injecting each
  // fault bit; force coincident internal DONE offers to check suppression.
  for(integer cause=0;cause<8;cause=cause+1)begin
   @(negedge clk);reset=1;repeat(4)@(negedge clk);reset=0;
   launch(0,32'h00000100);wait(arvalid);
   @(negedge clk);
   force dut.causes=8'(1<<cause);
   force dut.core_done=1'b1;force dut.dma_done=1'b1;force dut.header_status=1'b1;
   #1;
   if(core_done || ddone || hsvalid || !core_error)
    $fatal(1,"FAIL FAULT_BOUNDARY published completion cause=%0d",cause);
   release dut.core_done;release dut.dma_done;release dut.header_status;
   @(negedge clk);release dut.causes;
   if(!halted || !arvalid)$fatal(1,"FAIL FAULT_BOUNDARY stop/AR cause=%0d",cause);
   send_read();restore();healthy_read();
  end
  $display("PASS DATAPATH_FAULT_BOUNDARY injected_causes=8 coincident_completion_rejected=8 accepted_ar_drained=8 recovery_reads=8");
'''
    body=body.replace(marker,injection+marker)
   dst.write_text(body)
  else:shutil.copy2(f,dst)
  records.append(dict(path=str(f),source_sha256=sha(f),snapshot=str(dst),sha256=sha(dst)))
 env=os.environ.copy();env.update(PATH=str(BIN)+';E:/GIT/Git/usr/bin;'+env['PATH'],TEMP=str(root),TMP=str(root),TMPDIR=str(root),MSYS2_ARG_CONV_EXCL='*',VERILATOR_ROOT=str(VROOT),MAKE=str(BIN/'mingw32-make.exe'))
 top='tb_cnn_datapath_guard';obj=root/'obj'
 args=[BIN/'verilator_bin.exe','--cc','--exe','--main','--timing','--sched-zero-delay','--top-module',top,'-GACTIVE_TEST='+str(int(a.active)),'-GADDRESS_FAULT_TEST='+str(int(a.address_fault)),'--timescale-override','1ns/1ps','-Wno-fatal','--Mdir',obj.as_posix(),*[Path(r['snapshot']).as_posix() for r in records]]
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
 address_lines=re.findall(r'^PASS DATAPATH_ADDRESS_(?:DRAIN|ACTIVE).*$',body,re.M)
 assert len(address_lines)==2*int(a.address_fault)
 fault_boundary_lines=re.findall(r'^PASS DATAPATH_FAULT_BOUNDARY.*$',body,re.M)
 assert len(fault_boundary_lines)==int(a.fault_boundary)
 for r in records:assert sha(r['path'])==r['source_sha256'] and sha(r['snapshot'])==r['sha256']
 result=dict(status='INTEGRATED_DATAPATH_GUARD_NATIVE_PASS',line=lines[0],active_lines=active_lines,address_lines=address_lines,fault_boundary_lines=fault_boundary_lines,seconds=seconds,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),four_state=False,production_scheduler=False,full_frame=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
except Exception as e:
 (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
