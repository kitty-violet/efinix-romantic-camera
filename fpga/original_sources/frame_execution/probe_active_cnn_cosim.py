"""Run the existing numeric/held-token/fault fixture through a real DPI kernel.

All 25 production modules match the current frozen board simulation views.
The fixture contains its own memories/stimulus: this validates active native
execution and time scheduling, not the external AXI/PHY partition of the board.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from run_units import ROOT, HERE, sha
from probe_board_dpi_cost import BIN, VR, SIM, run


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--build-only',action='store_true');p.add_argument('--run-only',action='store_true')
    a=p.parse_args();root=a.root.resolve()
    assert root.is_relative_to(ROOT/'tmp')
    env=os.environ.copy();env.update(PATH=str(BIN)+';E:/GIT/Git/usr/bin;'+env['PATH'],
        VERILATOR_ROOT=VR.as_posix(),MSYS2_ARG_CONV_EXCL='*',SALT_LICENSE_SERVER='E:/quatra/LR-195771_License.dat')
    ref=ROOT/'tmp/frame-execution/pruned-unit1/frozen'
    board=ROOT/'tmp/frame-execution/board-repair-preflight2/accelerated_board'
    if not a.run_only:
        assert not root.exists();root.mkdir();src=root/'source';src.mkdir();obj=root/'obj'
        original=json.loads((ref/'inputs.json').read_text())
        bm=json.loads((board/'inputs.json').read_text());byname={Path(x['snapshot']).name:x for x in bm['inputs']}
        sources=[];records=[]
        for row in original['inputs']:
            name=Path(row['snapshot']).name
            item=row if name=='tb_compute_column_cluster.sv' else byname[name]
            f=Path(item['snapshot']);assert sha(f)==item['sha256'];dst=src/name
            shutil.copy2(f,dst);sources.append(dst)
            records.append(dict(snapshot=str(dst),sha256=sha(dst),board_source=name!='tb_compute_column_cluster.sv'))
        vectors=ref/'vectors';dump=root/'readback.hex'
        verify=subprocess.list2cmdline([sys.executable,'-B',str(HERE.parent/'coefficient_store/validate_readback.py'),
            '--image-dir',str(vectors),'--readback',str(dump)])
        argv=['active_cnn','+VECTORS='+vectors.as_posix(),'+DUMP='+dump.as_posix(),'+VERIFY_COMMAND='+verify]
        (root/'inputs.json').write_text(json.dumps(dict(inputs=records,board_inputs_sha256=sha(board/'inputs.json'),
            vectors={x.name:sha(x) for x in vectors.iterdir() if x.is_file()},fixture_reference=str(ref),
            scope='active numeric/backpressure/cancel/error fixture; not external AXI/PHY or full board',
            four_state=False,compile_timeout=300,simulation_timeout=600),indent=2))
        cmd=[BIN/'verilator_bin.exe','--cc','--timing','--sched-zero-delay','--top-module','tb_compute_column_cluster',
             '--threads','1','--timescale-override','1ns/1ps','--output-split','15000','--output-split-cfuncs','500',
             '-Wno-fatal','--Mdir',obj,'-CFLAGS','-DVL_TIME_CONTEXT -fPIC',*sources]
        run(cmd,root,'verilate',env,120)
        log=(root/'verilate.log').read_text(errors='replace')
        assert not re.search(r'%Warning-(UNOPTFLAT|LATCH|IMPLICIT|SELRANGE)',log)
        make=[BIN/'mingw32-make.exe','-f','Vtb_compute_column_cluster.mk','-j4','Vtb_compute_column_cluster__ALL.a',
            'verilated.o','verilated_threads.o','verilated_timing.o','CXX=g++.exe','AR=ar.exe',
            'PERL=E:/GIT/Git/usr/bin/perl.exe','SHELL=E:/GIT/Git/usr/bin/sh.exe','VERILATOR_ROOT='+VR.as_posix(),
            'OPT_FAST=-O2','OPT_GLOBAL=-O2','CXXFLAGS=-std=gnu++20 -DVL_TIME_CONTEXT']
        run(make,obj,'build',env,300)
        cpp=r'''
#include "Vtb_compute_column_cluster.h"
#include "verilated.h"
#include <cstdio>
#include <chrono>
using U=unsigned long long;using Clock=std::chrono::steady_clock;
struct Fixture{VerilatedContext*c;Vtb_compute_column_cluster*m;U calls=0;double seconds=0;};
extern "C" __declspec(dllexport) void* active_create(){
const char*argv[]={ARGV};auto*f=new Fixture;f->c=new VerilatedContext;f->c->threads(1);f->c->commandArgs(ARGC,argv);
f->m=new Vtb_compute_column_cluster(f->c);return f;}
extern "C" __declspec(dllexport) U active_step(void*p,U t){auto*f=static_cast<Fixture*>(p);++f->calls;auto start=Clock::now();
f->c->time(t);do{f->m->eval();}while(f->m->eventsPending()&&f->m->nextTimeSlot()==t&&!f->c->gotFinish());
f->seconds+=std::chrono::duration<double>(Clock::now()-start).count();
return f->c->gotFinish()?~0ULL:f->m->nextTimeSlot();}
extern "C" __declspec(dllexport) void active_final(void*p){auto*f=static_cast<Fixture*>(p);f->m->final();
std::printf("ACTIVE_DPI_STATS calls=%llu native_seconds=%.9f sim_ps=%llu\n",f->calls,f->seconds,(U)f->c->time());std::fflush(stdout);
delete f->m;delete f->c;delete f;}
#ifdef STANDALONE
int main(){auto*p=active_create();U t=0;while(t!=~0ULL)t=active_step(p,t);active_final(p);return 0;}
#endif
'''.replace('ARGV',','.join(json.dumps(x) for x in argv)).replace('ARGC',str(len(argv)))
        (root/'active.cpp').write_text(cpp)
        common=[BIN/'g++.exe','-O2','-std=c++20','-DVL_TIME_CONTEXT','-I'+obj.as_posix(),'-I'+(VR/'include').as_posix(),
            '-I'+(VR/'include/vltstd').as_posix(),root/'active.cpp',obj/'Vtb_compute_column_cluster__ALL.a',
            obj/'verilated.o',obj/'verilated_threads.o',obj/'verilated_timing.o','-pthread','-latomic']
        run([*common,'-shared','-o',root/'active.dll'],root,'link_dll',env,90)
        run([*common,'-DSTANDALONE','-o',root/'active.exe'],root,'link_exe',env,90)
        (root/'tb.sv').write_text('''module tb;
timeunit 1ps;timeprecision 1ps;
import "DPI-C" function chandle active_create();
import "DPI-C" function longint unsigned active_step(chandle h,longint unsigned t);
import "DPI-C" function void active_final(chandle h);
chandle h;longint unsigned due;
initial begin
h=active_create();due=active_step(h,0);
while(due!=64'hffffffffffffffff)begin
 if(due<$time)$fatal(1,"Active time reversed");
 #(due-$time);due=active_step(h,$time);
end
active_final(h);$display("PASS ACTIVE_CNN_DPI_SCHEDULE");$finish;
end
endmodule
''')
        (root/'run.do').write_text('onerror {quit -code 1}\nonbreak {quit -code 1}\nrun -all\nquit -code 0\n')
        print('ACTIVE_CNN_BUILD_READY',flush=True)
    if a.build_only:return
    manifest=json.loads((root/'inputs.json').read_text())
    for x in manifest['inputs']:assert sha(x['snapshot'])==x['sha256']
    times={}
    times['native']=run([root/'active.exe'],root,'native',env,180)
    run([SIM/'vlib.exe','work'],root,'vlib',env,20)
    run([SIM/'vlog.exe','-sv',root/'tb.sv'],root,'compile',env,30)
    times['dpi']=run([SIM/'vsim.exe','-c','-voptargs=-O5','-sv_lib',root/'active','work.tb','-do','run.do'],root,'simulation',env,600)
    def rows(name):
        text=(root/(name+'.log')).read_text(errors='replace')
        assert 'PASS COMPUTE_CLUSTER' in text and 'ACTUAL_RTL_READBACK_SOFTWARE_VERIFIED' in text
        assert not re.search(r'\bFAIL\b|%Error|\*\* (Fatal|Error)',text)
        return re.findall(r'(?:^|# )(BLOCK case=.*|PASS COMPUTE_CLUSTER.*|ACTIVE_DPI_STATS.*)$',text,re.M)
    nr=rows('native');dr=rows('simulation')
    assert nr[:-1]==dr[:-1] and len(nr)==26,(len(nr),len(dr))
    result=dict(status='ACTIVE_CNN_NATIVE_DPI_NUMERIC_CYCLE_MATCH_PASS',times=times,rows=dr,
        external_axi_partition=False,full_board=False,four_state=False,inputs_sha256=sha(root/'inputs.json'),
        native_log_sha256=sha(root/'native.log'),dpi_log_sha256=sha(root/'simulation.log'))
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
