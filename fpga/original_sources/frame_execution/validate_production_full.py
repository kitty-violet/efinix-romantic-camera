"""Build/replay one continuous RTL graph. No local timing extrapolation."""
import argparse,collections,json,os,re,shutil,subprocess,sys,traceback,zlib
from pathlib import Path
from run_units import HERE,ROOT,EF,sha
from validate_fourway import selected
sys.path.insert(0,str(HERE.parent/'pw_array_feed_abc'))
from run_native import BIN,VROOT,run
RUNNER_CONTENT=Path(__file__).read_bytes()

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--vectors',type=Path,required=True);p.add_argument('--reuse',type=Path);p.add_argument('--limit',type=int);p.add_argument('--optimized',action='store_true');p.add_argument('--production',action='store_true');p.add_argument('--dma-production',action='store_true');p.add_argument('--factored-dispatch',action='store_true');p.add_argument('--multi-issue',action='store_true');p.add_argument('--concurrent',action='store_true');p.add_argument('--profile',type=int,default=0);p.add_argument('--opt',choices=['O1','O2','O3'],default='O2');p.add_argument('--simulation-timeout',type=int,default=3600);p.add_argument('--rotating-dispatch',action='store_true');p.add_argument('--overrides-dir',type=Path);p.add_argument('--datapath',action='store_true');p.add_argument('--frozen-base',type=Path);p.add_argument('--descriptors',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False);vec=a.vectors.resolve()
 (root/'frozen_runner.py').write_bytes(RUNNER_CONTENT)
 try:
  vm=json.loads((vec/'frame_manifest.json').read_text())
  if a.limit is None:a.limit=vm['commands']
  assert 0<a.limit<=vm['commands']
  concurrency=vm.get('concurrency',{})
  if a.concurrent:assert a.optimized and concurrency['compute_commands']+concurrency['dma_commands']==vm['commands']
  for n,h in vm['files'].items():assert sha(vec/n)==h
  env=os.environ.copy();env.update(PATH=str(BIN)+';E:\\GIT\\Git\\usr\\bin;'+env['PATH'],TEMP=str(root),TMP=str(root),TMPDIR=str(root),
    MSYS2_ARG_CONV_EXCL='*',VERILATOR_ROOT=str(VROOT),MAKE=str(BIN/'mingw32-make.exe'))
  if a.reuse:
   prior=json.loads((a.reuse/'inputs.json').read_text());binary=Path(prior['binary']);records=prior['inputs']
   assert (prior['optimized'],prior.get('concurrent',False))==(a.optimized,a.concurrent),'Compiled mode differs from requested mode'
   assert prior.get('production',False)==a.production,'Compiled production mode differs'
   assert prior.get('dma_production',False)==a.dma_production,'Compiled DMA variant differs'
   assert prior.get('factored_dispatch',False)==a.factored_dispatch,'Compiled dispatcher variant differs'
   assert prior.get('multi_issue',False)==a.multi_issue,'Compiled reader variant differs'
   assert prior.get('rotating_dispatch',False)==a.rotating_dispatch,'Compiled rotating dispatcher differs'
   assert prior.get('datapath',False)==a.datapath,'Compiled datapath integration differs'
   assert prior.get('overrides_dir')==(str(a.overrides_dir.resolve()) if a.overrides_dir else None),'Compiled override differs'
   assert prior.get('frozen_base')==(str(a.frozen_base.resolve()) if a.frozen_base else None),'Compiled base snapshot differs'
   oldvm=json.loads((Path(prior['vectors'])/'frame_manifest.json').read_text())
   if (prior['limit']<oldvm['commands'])!=(a.limit<vm['commands']):
    tb=next(r for r in records if Path(r['snapshot']).name=='tb_fourway_full.sv')
    canonical=HERE/'variants/production_issue/tb_fourway_full.sv'
    assert tb.get('source_sha256')==sha(canonical),'Compiled smoke/full boundary differs'
    boundary=canonical.read_text()
    assert re.search(r'\n  end else begin\n(?:  \$display\([^\n]*\);\n)?  if\(error \|\| !program_frame_valid',boundary),'Missing mutually exclusive runtime boundary'
    assert 'wait(program_frame_valid || error || (limit<COMMANDS && commands_done>=limit));' in boundary,'Full graph must wait for actual frame publication'
   assert sha(Path(prior['vectors'])/'frame_manifest.json')==prior['vector_manifest_sha256']
   assert (oldvm['commands'],oldvm['expected_compute_words'],oldvm.get('concurrency',{}))==(vm['commands'],vm['expected_compute_words'],vm.get('concurrency',{})),'Compiled vector dimensions differ'
   assert sha(binary)==prior['binary_sha256']
   for r in records:assert sha(r['snapshot'])==r['sha256']
   warnings=prior['warning_codes']
  else:
   assert a.frozen_base is None or a.overrides_dir is not None,'Frozen base requires an explicit candidate override directory'
   src=root/'source';src.mkdir()
   paths=[EF/'sim_models/verilog'/n for n in ('efx_dsp24.v','efx_dsp48.v','efx_srl8.v','efx_ram10.v','efx_dpram10.v')]+selected(a.optimized,production=a.production)+[
    HERE/n for n in ('tensor_arena12.sv','resident_weight_bank64.sv','next_header_prefetch.sv','resident_read_arbiter2.sv','arena_client_router2.sv',
     'variants/production_banked/frame_arena_row_dma.sv' if a.dma_production else 'variants/throughput/frame_arena_row_dma.sv' if a.optimized else 'frame_arena_row_dma.sv','frame_execution_meter.sv','tb_fourway_full.sv')]+[
    HERE.parent/'full_frame_closure/coefficient_bank_store_fast.sv']
   if a.production:paths=[HERE/'variants/production_banked'/f.name if f.name=='tb_fourway_full.sv' else f for f in paths]
   if a.factored_dispatch:
    assert a.production
    paths=[HERE/'variants/factored_dispatch'/f.name if f.name=='unified_column_dispatch32.sv' else f for f in paths]
   if a.multi_issue:
    assert a.production
    paths=[HERE/'variants/multi_issue_reader'/f.name if f.name=='unified_column_reader.sv' else f for f in paths]
   if a.rotating_dispatch:
    assert a.production
    paths=[HERE/'variants/rotating_dispatch'/f.name if f.name=='unified_column_dispatch32.sv' else f for f in paths]
   if a.datapath:
    assert a.production and a.dma_production and a.concurrent
    paths=[HERE/'variants/integrated_datapath'/f.name if f.name in ('frame_arena_row_dma.sv','tb_fourway_full.sv') else f for f in paths]+[HERE/'cnn_frame_datapath.sv']
   if a.overrides_dir:
    from frozen_sources import frozen_paths
    paths=frozen_paths(paths,a.frozen_base)
    override=a.overrides_dir.resolve();assert override.is_relative_to(HERE/'variants')
    paths=[override/f.name if (override/f.name).is_file() else f for f in paths]
   paths=[HERE/'variants/production_issue/tb_fourway_full.sv' if f.name=='tb_fourway_full.sv' else f for f in paths]
   paths += [HERE/n for n in ('frame_program_fetch.sv','frame_program_issue.sv','axi_read_merge2.sv','cnn_production_frame.sv')]
   if a.frozen_base:
    paths=frozen_paths(paths,a.frozen_base)
    if a.overrides_dir:paths=[override/f.name if (override/f.name).is_file() else f for f in paths]
   paths=list(dict.fromkeys(paths))
   records=[]
   for f in paths:
    dst=src/f.name
    if f.name=='tb_fourway_full.sv':
     body=f.read_text()
     if a.overrides_dir and (override/'production_trace.svh').is_file():
      body=body.replace('endmodule',(override/'production_trace.svh').read_text()+'\nendmodule')
     dst.write_text(body)
    else:shutil.copy2(f,dst)
    records.append(dict(path=str(f),source_sha256=sha(f),snapshot=str(dst),sha256=sha(dst)))
   obj=root/'obj';top='tb_fourway_full'
   args=[BIN/'verilator_bin.exe','--cc','--exe','--main','--timing','--sched-zero-delay','--top-module',top,
    '-GOPTIMIZED='+str(int(a.optimized)),'-GCOMMANDS='+str(vm['commands']),'-GEXPECTED_WORDS='+str(vm['expected_compute_words']),
    '-GCONCURRENT='+str(int(a.concurrent)),'-GCOMPUTE_COMMANDS='+str(concurrency.get('compute_commands',0)),'-GDMA_COMMANDS='+str(concurrency.get('dma_commands',0)),
    '--timescale-override','1ns/1ps','--output-split','15000','--output-split-cfuncs','500','-Wno-fatal','--Mdir',obj.as_posix(),*[Path(r['snapshot']).as_posix() for r in records]]
   (root/'compile-arguments.json').write_text(json.dumps(list(map(str,args)),indent=2))
   body,sec=run(args,root,root/'compile.log',env)
   warnings=collections.Counter(re.findall(r'^%Warning(?:-([A-Z0-9_]+))?:',body,re.M))
   assert not(set(warnings)&{'UNOPTFLAT','LATCH','SELRANGE','IMPLICIT'}),warnings
   opts='-'+a.opt+' -munroll-only-small-loops -mvzeroupper'
   make=[BIN/'mingw32-make.exe','-f','V'+top+'.mk','-j4',f'CXX={(BIN/"g++.exe").as_posix()}',f'LINK={(BIN/"g++.exe").as_posix()}',f'AR={(BIN/"ar.exe").as_posix()}',
    'PERL=E:/GIT/Git/usr/bin/perl.exe',f'PYTHON3={Path(sys.executable).as_posix()}','SHELL=E:/GIT/Git/usr/bin/sh.exe',f'VERILATOR_ROOT={VROOT.as_posix()}',
    f'OPT_FAST={opts}',f'OPT_GLOBAL={opts}','CXXFLAGS=-std=gnu++20 -Werror=invalid-pch']
   (root/'make-arguments.json').write_text(json.dumps(list(map(str,make)),indent=2));run(make,obj,root/'build.log',env)
   binary=obj/('V'+top+'.exe')
  manifest=dict(frozen_base=str(a.frozen_base.resolve()) if a.frozen_base else None,inputs=records,binary=str(binary),binary_sha256=sha(binary),warning_codes=dict(warnings),vectors=str(vec),
    vector_manifest_sha256=sha(vec/'frame_manifest.json'),runner_sha256=sha(root/'frozen_runner.py'),datapath=a.datapath,profile=a.profile,limit=a.limit,opt=a.opt,simulation_timeout=a.simulation_timeout,optimized=a.optimized,production=a.production,dma_production=a.dma_production,factored_dispatch=a.factored_dispatch,multi_issue=a.multi_issue,rotating_dispatch=a.rotating_dispatch,overrides_dir=str(a.overrides_dir.resolve()) if a.overrides_dir else None,concurrent=a.concurrent)
  (root/'inputs.json').write_text(json.dumps(manifest,indent=2))
  checker=HERE.parent/'coefficient_store/validate_readback.py';dump=root/'readback.hex'
  command=subprocess.list2cmdline([sys.executable,'-B',str(checker),'--image-dir',str(vec),'--readback',str(dump)])
  final=next(n['offset'] for n in vm['nodes'] if n['id']=='rgb_head')
  descriptors=a.descriptors.resolve()
  dm=json.loads((descriptors/'result.json').read_text())
  for name in ('compute','dma'):assert sha(descriptors/(name+'.bin'))==dm['streams'][name]['sha256']
  crcargs=['+DESCRIPTORS='+descriptors.as_posix(),'+CCRC='+format(zlib.crc32((descriptors/'compute.bin').read_bytes()),'x'),'+DCRC='+format(zlib.crc32((descriptors/'dma.bin').read_bytes()),'x')]
  body,secs=run([binary,*crcargs,'+VECTORS='+vec.as_posix(),'+DUMP='+dump.as_posix(),'+VERIFY_COMMAND='+command,
    '+FINAL_GOLD_OFFSET='+str(final),'+PROFILE='+str(a.profile),'+LIMIT='+str(a.limit)],root,root/'simulation.log',env,timeout=a.simulation_timeout)
  assert 'ACTUAL_RTL_READBACK_SOFTWARE_VERIFIED' in body and not re.search(r'\bFAIL\b|%Error',body)
  rows=re.findall(r'^PASS CONTINUOUS_720P profile=(\d+) commands=(\d+) cycles=(\d+) scalar_checks=(\d+) final_bytes=(\d+) readbeats=(\d+) writebeats=(\d+) write_responses=(\d+) nodes=([0-9a-f]+)$',body,re.M)
  smoke=a.limit<vm['commands']
  if smoke:assert 'SMOKE FULL_CHAIN' in body and not rows;cycles=None
  else:assert len(rows)==1;cycles=int(rows[0][2]);assert int(rows[0][3])==vm['expected_compute_words']*8 and int(rows[0][4])==2764800
  smoke_rows=re.findall(r'^SMOKE FULL_CHAIN commands=(\d+) compute=(\d+) dma=(\d+) checks=(\d+) cycles=(\d+) visited_nodes=([0-9a-f]+) header_overlap=(\d+)$',body,re.M)
  if smoke:assert len(smoke_rows)==1 and int(smoke_rows[0][0])>=a.limit
  result=dict(status='CONTINUOUS_FRAME_SMOKE_ONLY' if smoke else 'CONTINUOUS_720P_RTL_NUMERIC_MEASUREMENT_PASS',profile=a.profile,commands=a.limit,cycles=cycles,smoke_measurements=smoke_rows,
    measured_frame_ms=cycles/150000 if cycles else None,clock_hz_assumed=150000000,
    reasons=[dict(id=int(i),cycles=int(c)) for i,c in re.findall(r'^REASON id=(\d+) cycles=(\d+)$',body,re.M)],
    nodes=[dict(id=int(i),cycles_with_header_control=int(c),words=int(w)) for i,c,w in re.findall(r'^NODE id=(\d+) cycles_with_header_control=(\d+) words=(\d+)$',body,re.M)],
    stalls=[dict(zip(('node','input','weight','pool','dispatch','setup_drain','owner_accepts'),map(int,row))) for row in re.findall(r'^STALL node=(\d+) input=(\d+) weight=(\d+) pool=(\d+) dispatch=(\d+) setup_drain=(\d+) owner_accepts=(\d+)$',body,re.M)],
    overlaps=[dict(header_compute=int(h),simultaneous_resident_issue=int(s)) for h,s in re.findall(r'^OVERLAP header_compute=(\d+) simultaneous_resident_issue=(\d+)$',body,re.M)],
    input_causes=[dict(zip(('node','empty','offer','return','other'),map(int,row))) for row in re.findall(r'^INPUT_CAUSE node=(\d+) empty=(\d+) offer=(\d+) return=(\d+) other=(\d+)$',body,re.M)],
    offer_causes=[dict(zip(('node','skip','writer','external','accepted_now','internal'),map(int,row))) for row in re.findall(r'^OFFER_CAUSE node=(\d+) skip=(\d+) writer=(\d+) external=(\d+) accepted_now=(\d+) internal=(\d+)$',body,re.M)],
    dma_overlap=[dict(zip(('active','hidden','compute_dependency_wait','dma_dependency_wait'),map(int,row))) for row in re.findall(r'^DMA_OVERLAP active=(\d+) hidden=(\d+) compute_dependency_wait=(\d+) dma_dependency_wait=(\d+)$',body,re.M)],
    line=rows[0] if rows else None,simulation_seconds=secs,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),
    complete_model_graph=not smoke,frame_30fps_measured=bool(cycles and cycles<=5000000),full_baseline=False,production_program=True,descriptor_manifest_sha256=sha(descriptors/'result.json'),
    scheduling='actual finite hardware DDR descriptor fetch and issuer, CRC and final RGB publication; host coefficient loading is testbench driven',ddr='AXI behavioral memory with actual shared program/data reader; no PHY/refresh/other system masters',
    weights='synthetic diagnostic',physical_timing_pass=False,board_test=False,steady_frame_interval_measured=False)
  for r in records:assert sha(r['snapshot'])==r['sha256']
  assert sha(binary)==manifest['binary_sha256']
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()



