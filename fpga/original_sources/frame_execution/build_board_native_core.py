"""Native compilation of the exact mapped CNN RTL for Questa DPI co-simulation.

This changes simulator execution only. It is not a replacement algorithm or a
hardware optimization. The caller must guard four-state input validity.
"""
import argparse, json, os, re, shutil, subprocess, sys, time
from pathlib import Path
from run_units import ROOT, HERE, EF, sha

def run(args, cwd, log, env):
    start=time.monotonic()
    with log.open('wb') as f:
        subprocess.run(list(map(str,args)),cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    return time.monotonic()-start

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--reuse-native',type=Path);p.add_argument('--threads',type=int,default=4,choices=(1,4));p.add_argument('--packed-bridge',action='store_true');a=p.parse_args()
    root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    root.mkdir();src=root/'source';src.mkdir();base=a.base.resolve()
    bm=json.loads((base/'inputs.json').read_text()); byname={Path(r['snapshot']).name:r for r in bm['inputs']}
    prior=json.loads((ROOT/'tmp/frame-execution/video-pipeline-frame8/inputs.json').read_text())
    records=[]
    for old in prior['inputs']:
        name=Path(old['snapshot']).name
        if name in ('tb_fourway_full.sv','frame_execution_meter.sv'):continue
        if name in byname:
            r=byname[name];f=Path(r['snapshot']);assert sha(f)==r['sha256']
        else:
            f=EF/'sim_models/verilog'/name
            assert f.is_file() and name.startswith('efx_'),name
        dst=src/name;shutil.copy2(f,dst);records.append(dict(source=str(f),snapshot=str(dst),sha256=sha(dst)))
    shutil.copy2(base/'ddr_memory_map.vh',src/'ddr_memory_map.vh')
    for f in base.rglob('*.vh'):
        if not (src/f.name).exists():shutil.copy2(f,src/f.name)
    bin=ROOT/'tools/verilator/ucrt64/bin';vr=ROOT/'tools/verilator/ucrt64/share/verilator'
    env=os.environ.copy();env.update(PATH=str(bin)+';E:/GIT/Git/usr/bin;'+env['PATH'],VERILATOR_ROOT=vr.as_posix(),MSYS2_ARG_CONV_EXCL='*')
    # The standard lib-create bridge cannot host timed vendor RAM primitives.
    # Reuse its generated DPI interface with a timestamp, while compiling the
    # RTL normally with timing and delta scheduling intact.
    obj=root/'obj';args=[bin/'verilator_bin.exe','--cc','--timing','--sched-zero-delay','--top-module','cnn_video_pipeline',
        '-GCPU_DDR_BASE=67108864','-I'+src.as_posix(),'--timescale-override','1ns/1ps','--threads',str(a.threads),
        '--output-split','15000','--output-split-cfuncs','500','-Wno-fatal','--Mdir',obj.as_posix(),
        '-CFLAGS','-DVL_TIME_CONTEXT -fPIC',*[r['snapshot'] for r in records]]
    (root/'compile_args.json').write_text(json.dumps(list(map(str,args)),indent=2))
    if a.reuse_native:
        prior=json.loads((a.reuse_native/'inputs.json').read_text())
        assert prior.get('threads',4)==a.threads
        assert {Path(r['snapshot']).name:r['sha256'] for r in prior['inputs']}=={Path(r['snapshot']).name:r['sha256'] for r in records}
        shutil.copytree(a.reuse_native/'obj',obj);shutil.copy2(a.reuse_native/'compile.log',root/'compile.log');seconds=0
    else:seconds=run(args,root,root/'compile.log',env)
    log=(root/'compile.log').read_text(errors='replace')
    warnings=dict(__import__('collections').Counter(re.findall(r'^%Warning(?:-([A-Z0-9_]+))?:',log,re.M)))
    assert not (set(warnings)&{'UNOPTFLAT','LATCH','SELRANGE','IMPLICIT','STMTDLY'}),warnings
    interface=ROOT/'tmp/frame-execution/board-native-core1/obj'
    cpp=(interface/'boardcnn.cpp').read_text().replace('long long m_seqnum;','long long m_seqnum = 0;')
    cpp=cpp.replace('Vcnn_video_pipeline_container* const handlep__V = new',f'Verilated::threadContextp()->threads({a.threads});\n        Vcnn_video_pipeline_container* const handlep__V = new')
    cpp=cpp.replace('void* vhandlep__V,','void* vhandlep__V, unsigned long long time_ps,')
    cpp=cpp.replace('handlep__V->eval();','''if (handlep__V->m_seqnum < 20 || handlep__V->m_seqnum % 100000 == 0) {
            fprintf(stderr,"DPI_TIME seq=%lld target=%llu current=%llu pending=%d next=%llu\\n",handlep__V->m_seqnum,time_ps,
                    (unsigned long long)handlep__V->contextp()->time(),handlep__V->eventsPending(),
                    handlep__V->eventsPending()?(unsigned long long)handlep__V->nextTimeSlot():0);fflush(stderr);
        }
        handlep__V->contextp()->time(time_ps);
        do { handlep__V->eval(); }
        while (handlep__V->eventsPending() && handlep__V->nextTimeSlot() == time_ps);''')
    cpp+='''
extern "C" unsigned long long boardcnn_next_time(void* p) {
 auto* m=static_cast<Vcnn_video_pipeline_container*>(p);
 return m->eventsPending()?m->nextTimeSlot():~0ULL;
}
'''
    packed_sv=None;packed_info=None
    if a.packed_bridge:
        from board_integration.native_packed_bridge import generate
        cpp,packed_sv,packed_info=generate(interface/'boardcnn.sv',(src/'cnn_video_pipeline.sv').read_text().split(') (',1)[0])
    (obj/'boardcnn.cpp').write_text(cpp)
    make=[bin/'mingw32-make.exe','-f','Vcnn_video_pipeline.mk','-j4','Vcnn_video_pipeline__ALL.a',
          'boardcnn.o','verilated.o','verilated_threads.o','verilated_timing.o',
          'CXX=g++.exe','LINK=g++.exe','AR=ar.exe','PERL=E:/GIT/Git/usr/bin/perl.exe',
          'PYTHON3='+Path(sys.executable).as_posix(),'SHELL=E:/GIT/Git/usr/bin/sh.exe',
          'VERILATOR_ROOT='+vr.as_posix(),'OPT_FAST=-O2 -munroll-only-small-loops -mvzeroupper',
          'OPT_GLOBAL=-O2','CXXFLAGS=-std=gnu++20 -DVL_TIME_CONTEXT']
    manifest=dict(base=str(base),base_inputs_sha256=sha(base/'inputs.json'),inputs=records,warning_codes=warnings,
                  simulator='Verilator native RTL linked by DPI to Questa',two_state_core=True,physical_pass=False,threads=a.threads,packed_bridge=packed_info)
    (root/'inputs.json').write_text(json.dumps(manifest,indent=2))
    seconds+=run(make,obj,root/'build.log',env)
    dll=obj/'libboardcnn.dll'
    seconds+=run([bin/'g++.exe','-shared','-o',dll,'boardcnn.o','-Wl,--whole-archive','Vcnn_video_pipeline__ALL.a',
                  '-Wl,--no-whole-archive','verilated.o','verilated_threads.o','verilated_timing.o','-pthread','-latomic'],obj,root/'link.log',env)
    # Keep the production module interface/parameters. Native elaboration is
    # fixed to the board values; reject other parameterizations at time zero.
    original=(src/'cnn_video_pipeline.sv').read_text();params=original.split(') (',1)[0]
    wrapper=(interface/'boardcnn.sv').read_text().replace('module boardcnn (',params+') (',1)
    wrapper=wrapper.replace('    // Precision of submodule','    timeunit 1ps; timeprecision 1ps;\n    // Precision of submodule',1)
    wrapper=wrapper.replace('chandle handle__V\n        ,','chandle handle__V, input longint unsigned time_ps\n        ,')
    wrapper=wrapper.replace('            handle__V,','            handle__V, $time,')
    wrapper=wrapper.replace('always @(cancel or reset or process_clk or display_clk)',
                            'always @(cancel or reset or process_clk or display_clk or native_tick)')
    wrapper=wrapper.replace('    final boardcnn_protectlib_final', '''
    import "DPI-C" function longint unsigned boardcnn_next_time(chandle handle);
    reg native_tick=0;
    longint unsigned native_due, native_scheduled=64'hffffffffffffffff;
    longint unsigned native_wakeup=64'hffffffffffffffff;
    // Transport the due timestamp through one scheduled assignment. Old
    // notifications can safely cause an extra eval; never cancel/recreate SV
    // processes on every input transition.
    always @(last_combo_seqnum__V or last_seq_seqnum__V) begin
        native_due=boardcnn_next_time(handle__V);
        if(native_due!=64'hffffffffffffffff && native_due!=native_scheduled)begin
            if(native_due<$time)$fatal(1,"Native timer went backwards");
            native_scheduled=native_due;
            native_wakeup <= #(native_due-$time) native_due;
        end
    end
    always @(native_wakeup) if(native_wakeup==$time) native_tick=~native_tick;
    final boardcnn_protectlib_final''',1)
    if packed_sv:
        timer=wrapper[wrapper.index('    reg native_tick=0;'):wrapper.index('    final boardcnn_protectlib_final')]
        wrapper=packed_sv+timer+'\nendmodule\n'
    checks="\ninitial begin\n"
    expected={'CPU_DDR_BASE':67108864,'CPU_ADDRESS_BASE':0xe1000000,'CPU_ADDRESS_LIMIT':0xe1ffffff,
              'INPUT_WIDTH':1920,'INPUT_HEIGHT':1080,'PROCESS_WIDTH':1280,'PROCESS_HEIGHT':720,
              'MAX_CAPTURE_BURST':1,'OUTPUT_FIFO_DEPTH':4096,'OUTPUT_PREFILL_PIXELS':3840,
              'FRAME_ID_WIDTH':16,'RESET_GUARD_CYCLES':20,'H_FRONT_PORCH':88,'H_SYNC':44,'H_BACK_PORCH':148,
              'V_FRONT_PORCH':4,'V_SYNC':5,'V_BACK_PORCH':36,'HS_POLARITY':1,'VS_POLARITY':1}
    for name,value in expected.items():checks+=f' if({name} != 32\'h{value:08x}) $fatal(1,"Native parameter mismatch: {name}");\n'
    wrapper=wrapper.replace('endmodule',checks+'end\nendmodule',1)
    (root/'cnn_video_pipeline_dpi.sv').write_text(wrapper)
    result=dict(status='NATIVE_LIBRARY_BUILT_NOT_YET_FRAME_VALIDATED',seconds=seconds,dll=str(dll),dll_sha256=sha(dll),
                inputs_sha256=sha(root/'inputs.json'),wrapper_sha256=sha(root/'cnn_video_pipeline_dpi.sv'))
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
