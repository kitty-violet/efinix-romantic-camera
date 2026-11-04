"""Bounded bridge cost attribution; this is not an active-CNN or board PASS."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
from run_units import ROOT, sha
from owned_simulator_process import spawn_owned, stop_owned_tree, close_owned

BIN = ROOT/'tools/verilator/ucrt64/bin'
VR = ROOT/'tools/verilator/ucrt64/share/verilator'
SIM = Path('E:/quatra/questa_fse/win64')


def run(command, root, name, env, timeout):
    start = time.perf_counter()
    with (root/(name+'.log')).open('w', encoding='utf8') as log:
        proc = spawn_owned(list(map(str, command)), cwd=root, env=env,
                           stdout=log, stderr=subprocess.STDOUT)
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            stop_owned_tree(proc)
            proc.wait(timeout=15)
            raise
        finally:
            close_owned(proc)
    assert code == 0, (name, code)
    return time.perf_counter()-start


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--coalesced', action='store_true')
    p.add_argument('--compare', action='store_true')
    p.add_argument('--unified', action='store_true')
    p.add_argument('--lazy-internal', action='store_true')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--batch', action='store_true')
    p.add_argument('--nosyncio', action='store_true')
    a=p.parse_args()
    root=a.root.resolve(); native=a.native.resolve()
    assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    root.mkdir()
    env=os.environ.copy()
    env.update(PATH=str(BIN)+';'+env['PATH'], SALT_LICENSE_SERVER='E:/quatra/LR-195771_License.dat')
    library=native/'obj/libboardcnn.dll'
    # Wrap the existing immutable DLL. No model recompilation or changes.
    cpp=r'''
#include <windows.h>
#include <cstdio>
#include <cstdint>
#include <chrono>
using U=unsigned long long;
using Clock=std::chrono::steady_clock;
static HMODULE mod;
static void*(*create_fn)(); static void(*final_fn)(void*);
static U(*combo_fn)(void*,U,const unsigned*,unsigned*);
static U(*seq_fn)(void*,U,const unsigned*,unsigned*);
static U(*next_fn)(void*);static void(*ignore_fn)(void*,const unsigned*);
static U combos=0,seqs=0,nexts=0,ignores=0;static double seconds=0;
template<class F>void symbol(F&f,const char*n){f=reinterpret_cast<F>(GetProcAddress(mod,n));if(!f){std::fprintf(stderr,"MISSING %s\n",n);std::abort();}}
extern "C" __declspec(dllexport) void* boardcnn_packed_create(){
mod=LoadLibraryA("LIBPATH");if(!mod){std::fprintf(stderr,"LOAD_ERROR %lu\n",GetLastError());std::abort();}
symbol(create_fn,"boardcnn_packed_create");symbol(final_fn,"boardcnn_packed_final");
symbol(combo_fn,"boardcnn_packed_combo");symbol(seq_fn,"boardcnn_packed_seq");
symbol(next_fn,"boardcnn_next_time");symbol(ignore_fn,"boardcnn_packed_ignore");return create_fn();}
extern "C" __declspec(dllexport) void boardcnn_probe_report(int stage){
std::printf("BRIDGE_COST stage=%d combo=%llu seq=%llu next=%llu ignore=%llu native_seconds=%.9f\n",stage,combos,seqs,nexts,ignores,seconds);std::fflush(stdout);}
extern "C" __declspec(dllexport) void boardcnn_packed_final(void*p){boardcnn_probe_report(9);final_fn(p);FreeLibrary(mod);}
extern "C" __declspec(dllexport) U boardcnn_packed_combo(void*p,U t,const unsigned*i,unsigned*o){++combos;auto a=Clock::now();auto r=combo_fn(p,t,i,o);seconds+=std::chrono::duration<double>(Clock::now()-a).count();return r;}
extern "C" __declspec(dllexport) U boardcnn_packed_seq(void*p,U t,const unsigned*i,unsigned*o){++seqs;auto a=Clock::now();auto r=seq_fn(p,t,i,o);seconds+=std::chrono::duration<double>(Clock::now()-a).count();return r;}
extern "C" __declspec(dllexport) U boardcnn_next_time(void*p){++nexts;return next_fn(p);}
extern "C" __declspec(dllexport) void boardcnn_packed_ignore(void*p,const unsigned*i){++ignores;ignore_fn(p,i);}
'''.replace('LIBPATH', library.as_posix())
    cpp='#include <cstdlib>\n#include <cstring>\n#include <unordered_map>\n#include <array>\n'+cpp
    cpp+=r'''
extern "C" __declspec(dllexport) U boardcnn_fused_combo(void*p,U t,const unsigned*i,unsigned*o,U*due){auto r=boardcnn_packed_combo(p,t,i,o);*due=next_fn(p);return r;}
extern "C" __declspec(dllexport) U boardcnn_fused_seq(void*p,U t,const unsigned*all,const unsigned*i,unsigned*o,U*due){ignore_fn(p,all);auto r=boardcnn_packed_seq(p,t,i,o);*due=next_fn(p);return r;}
struct InputState{bool init=false;std::array<unsigned,27> in{};unsigned clock=0;std::array<unsigned,30> out{};};
static std::unordered_map<void*,InputState> states;
extern "C" __declspec(dllexport) U boardcnn_unified_step(void*p,U t,const unsigned*i,unsigned clock,unsigned*o){
auto&s=states[p];
LAZY_INTERNAL
if(!s.init||std::memcmp(i,s.in.data(),108)){boardcnn_packed_combo(p,t,i,s.out.data());std::memcpy(s.in.data(),i,108);}
if(!s.init||clock!=s.clock||next_fn(p)==t){boardcnn_packed_seq(p,t,&clock,s.out.data());s.clock=clock;}
s.init=true;std::memcpy(o,s.out.data(),120);return next_fn(p);
}
'''
    lazy=r'''
if(s.init){U due=next_fn(p);unsigned check[30];while(due<t){
boardcnn_packed_seq(p,due,&s.clock,check);
if(std::memcmp(check,s.out.data(),120)){std::fprintf(stderr,"LAZY_VISIBLE_EVENT due=%llu current=%llu\n",due,t);std::fflush(stderr);std::abort();}
due=next_fn(p);
}}
'''
    cpp=cpp.replace('LAZY_INTERNAL',lazy if a.lazy_internal else '')
    (root/'bridge_meter.cpp').write_text(cpp,encoding='utf8')
    run([BIN/'g++.exe','-std=c++20','-O2','-shared',root/'bridge_meter.cpp','-o',root/'bridge_meter.dll'],root,'build',env,90)
    original_wrapper=native/'cnn_video_pipeline_dpi.sv'
    original=original_wrapper.read_text()
    wrapper=root/'cnn_video_pipeline_dpi.sv'
    source=original
    if a.coalesced or a.compare:
        source=source.replace('import "DPI-C" function longint unsigned boardcnn_next_time(chandle h);','')
        source=source.replace('boardcnn_packed_combo(chandle h,longint unsigned t,input bit [858:0] data,output bit [950:0] result);',
            'boardcnn_fused_combo(chandle h,longint unsigned t,input bit [858:0] data,output bit [950:0] result,output longint unsigned due);')
        source=source.replace('boardcnn_packed_seq(chandle h,longint unsigned t,input bit [3:0] data,output bit [950:0] result);',
            'boardcnn_fused_seq(chandle h,longint unsigned t,input bit [858:0] all_data,input bit [3:0] data,output bit [950:0] result,output longint unsigned due);')
        source=source.replace('import "DPI-C" function void boardcnn_packed_ignore(chandle h,input bit [858:0] data);','')
        source=source.replace('chandle handle__V;', 'chandle handle__V;\nlongint unsigned combo_due,seq_due;')
        source=source.replace('boardcnn_packed_combo(handle__V,$time,packed_inputs,combo_out)',
                              'boardcnn_fused_combo(handle__V,$time,packed_inputs,combo_out,combo_due)')
        source=source.replace(' boardcnn_packed_ignore(handle__V,packed_inputs);','')
        source=source.replace('boardcnn_packed_seq(handle__V,$time,packed_clocks,tmp_out)',
                              'boardcnn_fused_seq(handle__V,$time,packed_inputs,packed_clocks,tmp_out,seq_due)')
        source=source.replace('native_due=boardcnn_next_time(handle__V);',
                             'native_due=(last_seq_seqnum__V>last_combo_seqnum__V)?seq_due:combo_due;')
    if a.unified:
        # One scheduled SV process and one DPI transition. Internal events keep
        # their original timestamps; no lookahead or omitted RAM delays.
        head_end=original.index('    );')+len('    );')
        prefix=original[:head_end]
        wires=original[original.index('wire [858:0] packed_inputs'):original.index('bit [950:0] combo_out')]
        output_assign=original[original.index('assign {loaded_words'):original.index('initial handle__V')]
        output_assign=output_assign[:output_assign.index('=')+1]+'published;\n'
        checks=original[original.rindex('\ninitial begin'):]
        source=prefix+'''
timeunit 1ps;timeprecision 1ps;
import "DPI-C" function chandle boardcnn_packed_create();
import "DPI-C" function void boardcnn_packed_final(chandle h);
import "DPI-C" function longint unsigned boardcnn_unified_step(chandle h,longint unsigned t,input bit [858:0] inputs,input int unsigned clocks,output bit [950:0] result);
chandle handle__V;
bit [950:0] published,temp;
reg native_tick=0;
longint unsigned due,scheduled=64'hffffffffffffffff,wakeup=64'hffffffffffffffff;
'''+wires+output_assign+'''
initial begin handle__V=boardcnn_packed_create();#0;native_tick=1;end
always @(packed_inputs or packed_clocks or native_tick) if(handle__V!=null)begin
 due=boardcnn_unified_step(handle__V,$time,packed_inputs,{28'd0,packed_clocks},temp);
 published<=temp;
 if(due!=64'hffffffffffffffff && due!=scheduled)begin
  if(due<$time)$fatal(1,"Unified native timer went backwards");
  scheduled=due;wakeup<=#(due-$time)due;
 end
end
always @(wakeup)if(wakeup==$time)native_tick=~native_tick;
final boardcnn_packed_final(handle__V);
'''+checks
        if a.lazy_internal:
            source=source.replace(" if(due!=64'hffffffffffffffff && due!=scheduled)begin", " if(1'b0)begin")
    wrapper.write_text(source)
    head=original.split(') (',1)[1].split(');',1)[0]
    ports=[(d,int(w)+1 if w else 1,n) for d,w,n in re.findall(r'\b(input|output) logic (?:\[(\d+):0\]\s+)?\s*(\w+)',head)]
    tb='`timescale 1ns/1ps\nmodule tb;\n'
    for d,w,n in ports:
        tb+=f'{"reg" if d=="input" else "wire"} [{w-1}:0] {n}'+('=0;\n' if d=='input' else ';\n')
    tb+='''
always #3.333 process_clk=~process_clk;
always #6.734 display_clk=~display_clk;
cnn_video_pipeline #(.CPU_DDR_BASE(32'h04000000)) dut(.*);
import "DPI-C" function void boardcnn_probe_report(int stage);
initial begin reset=1;#200;reset=0;end
initial begin #20000;boardcnn_probe_report(1);#80000;boardcnn_probe_report(2);$display("PASS BRIDGE_COST_ONLY_100US");$finish;end
endmodule
'''
    files=[wrapper]
    if a.compare:
        ref=root/'reference.sv'
        ref.write_text(original.replace('module cnn_video_pipeline #(', 'module cnn_video_pipeline_reference #(',1))
        assert 'module cnn_video_pipeline_reference' in ref.read_text()
        files.insert(0,ref)
        declarations=''; connections=[]; outputs=[]
        for d,w,n in ports:
            if d=='output':
                declarations+=f'wire [{w-1}:0] ref_{n};\n'
                connections.append(f'.{n}(ref_{n})');outputs.append((n,w))
            else:connections.append(f'.{n}({n})')
        declarations+='cnn_video_pipeline_reference #(.CPU_DDR_BASE(32\'h04000000)) ref_dut('+','.join(connections)+');\n'
        declarations+='integer checked=0;\nalways @(process_clk or display_clk) begin\n#0.003;\n'
        for n,w in outputs:declarations+=f'if ({n} !== ref_{n}) $fatal(1,"BRIDGE_OUTPUT_MISMATCH {n} t=%t",$time);\n'
        declarations+='checked=checked+1;\nend\n'
        # Exercise real coefficient-store load, bad-length recovery and video
        # handshakes, without pretending to have started CNN computation.
        stimulus='''
initial begin
 #1000; begin_version=1;begin_image_bytes=17792;begin_tag=16'h4321;
 @(negedge process_clk);begin_valid=1;
 while(!begin_ready) @(negedge process_clk);
 @(negedge process_clk);begin_valid=0;
 for(integer k=0;k<96;k=k+1)begin
  load_index=k;load_data=64'h1234567800000000+k;load_last=(k==95);load_valid=1;
  do @(posedge process_clk);while(!load_ready);
  @(negedge process_clk);load_valid=0;
  repeat(k%3)@(negedge process_clk);
 end
 #100;reset=1;#100;reset=0;
 repeat(200) begin @(negedge process_clk); s_rgb=s_rgb+24'h193751;s_valid=~s_valid; m_axi_awready=~m_axi_awready;m_axi_arready=~m_axi_arready; end
end
'''
        tb=tb.replace('endmodule', declarations+stimulus+'endmodule')
        tb=tb.replace('$display("PASS BRIDGE_COST_ONLY_100US")','$display("PASS BRIDGE_TRANSPORT_EQUIVALENCE checks=%0d",checked);$display("PASS BRIDGE_COST_ONLY_100US")')
    (root/'tb.sv').write_text(tb,encoding='utf8')
    (root/'run.do').write_text('onerror {quit -code 1}\nonbreak {quit -code 1}\nset t [clock milliseconds]\nrun 20 us\nputs "HOST_RUN20_MS [expr {[clock milliseconds]-$t}]"\nset t [clock milliseconds]\nrun -all\nputs "HOST_RUN80_MS [expr {[clock milliseconds]-$t}]"\nquit -code 0\n')
    if a.profile:
        (root/'run.do').write_text('onerror {quit -code 1}\nonbreak {quit -code 1}\nprofile on -p\nrun 50 us\nprofile summary -file profile-summary.txt\nprofile report -ranked -file profile-ranked.txt\nrun -all\nquit -code 0\n')
    manifest=dict(library=str(library),library_sha256=sha(library),wrapper=str(wrapper),wrapper_sha256=sha(wrapper),coalesced=a.coalesced,compare=a.compare,unified=a.unified,lazy_internal=a.lazy_internal,batch=a.batch,nosyncio=a.nosyncio,
        probe_sha256=sha(__file__),scope='idle cost attribution only, not active computation or full board',simulation_timeout=150)
    (root/'inputs.json').write_text(json.dumps(manifest,indent=2))
    run([SIM/'vlib.exe','work'],root,'vlib',env,20)
    run([SIM/'vlog.exe','-sv',*files,root/'tb.sv'],root,'compile',env,30)
    seconds=run([SIM/'vsim.exe','-batch' if a.batch else '-c',*(['-nosyncio'] if a.nosyncio else []),'-voptargs=-O5','-sv_lib',root/'bridge_meter','work.tb','-do','run.do'],root,'simulation',env,240 if a.compare else 150)
    log=(root/'simulation.log').read_text(encoding='utf8',errors='replace')
    assert 'PASS BRIDGE_COST_ONLY_100US' in log and not re.search(r'\*\* (Fatal|Error)',log)
    rows=re.findall(r'BRIDGE_COST stage=(\d+) combo=(\d+) seq=(\d+) next=(\d+) ignore=(\d+) native_seconds=([\d.]+)',log)
    assert len(rows)>=2
    if a.compare: assert 'PASS BRIDGE_TRANSPORT_EQUIVALENCE' in log
    result=dict(status='BRIDGE_COST_ATTRIBUTION_ONLY',wall_seconds=seconds,stages=rows,
        run20_ms=re.findall(r'HOST_RUN20_MS (\d+)',log),run80_ms=re.findall(r'HOST_RUN80_MS (\d+)',log),
        active_cnn=False,complete_board=False,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'))
    (root/'result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
