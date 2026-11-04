"""Build a narrow, cycle-causal production-CNN DPI boundary from frozen RTL.

Capture, display, shared arbitration, CPU and vendor PHY stay in Questa.
This builder does not apply speculative lookahead or change hardware sources.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
from probe_board_dpi_cost import BIN, VR, run
from run_units import ROOT, sha


def ports_of(source):
    header=source.split('module cnn_production_frame(',1)[1].split(');',1)[0]
    ports=[];direction=None;width=None
    for item in header.split(','):
        item=item.strip()
        m=re.fullmatch(r'(input|output)\s+wire\s*(?:\[(\d+):0\])?\s*(\w+)',item)
        if m:
            direction=m[1];width=int(m[2])+1 if m[2] else 1;name=m[3]
        else:
            assert direction and re.fullmatch(r'\w+',item),item
            name=item
        ports.append((direction,width,name))
    assert len(ports)==len({n for d,w,n in ports})
    return header,ports


def generate(source):
    header,ports=ports_of(source)
    controls={'clk','reset','cancel'}
    normal=[(n,w) for d,w,n in ports if d=='input' and n not in controls]
    clocks=[(n,w) for d,w,n in ports if n in controls]
    outputs=[(n,w) for d,w,n in ports if d=='output']
    taps=[('compute_begin',1),('compute_end',1),('debug_program_index',15)]
    def fields(items,read):
        code=[];pos=0
        for n,w in items:
            if read and w<=64:code.append(f'm->{n}=get(data,{pos},{w});')
            else:
                for k in range((w+31)//32):
                    count=min(32,w-k*32);v=f'm->{n}[{k}]' if w>64 else f'(uint64_t(m->{n})>>{k*32})'
                    code.append(f'm->{n}[{k}]=get(data,{pos+k*32},{count});' if read else f'put(out,{pos+k*32},{count},{v});')
            pos+=w
        return '\n'.join(code),pos
    combos,ni=fields(normal,True);seqs,nc=fields(clocks,True);publish,no=fields(outputs,False)
    for name,width in taps:
        publish+=f'\nput(out,{no},{width},m->rootp->cnn_production_frame__DOT__{name});'
        no+=width
    cpp=r'''
#include "Vcnn_production_frame.h"
#include "Vcnn_production_frame___024root.h"
#include "verilated.h"
#include <cstdint>
#include <cstring>
using U=unsigned long long;
struct Model:Vcnn_production_frame {U sequence=0;Model(VerilatedContext*c):Vcnn_production_frame(c){};};
static uint64_t get(const unsigned*a,unsigned pos,unsigned width){
 uint64_t out=0;unsigned done=0;
 while(done<width){unsigned s=pos%32,n=std::min(width-done,32-s);uint64_t mask=(1ULL<<n)-1;
 out|=((uint64_t(a[pos/32])>>s)&mask)<<done;done+=n;pos+=n;}return out;
}
static void put(unsigned*a,unsigned pos,unsigned width,uint64_t value){
 unsigned done=0;while(done<width){unsigned s=pos%32,n=std::min(width-done,32-s);
 a[pos/32]|=unsigned((value>>done)&((1ULL<<n)-1))<<s;done+=n;pos+=n;}
}
extern "C" __declspec(dllexport) void* production_create(){auto*c=new VerilatedContext;c->threads(1);return new Model(c);}
extern "C" __declspec(dllexport) void production_final(void*p){auto*m=static_cast<Model*>(p);auto*c=m->contextp();m->final();delete m;delete c;}
'''
    for kind,body in [('combo',combos),('seq',seqs)]:
        cpp+=f'''
extern "C" __declspec(dllexport) U production_{kind}(void*p,U t,const unsigned*data,unsigned*out,U*due){{
 auto*m=static_cast<Model*>(p);{body}
 m->contextp()->time(t);do{{m->eval();}}while(m->eventsPending()&&m->nextTimeSlot()==t);
 std::memset(out,0,{(no+31)//32*4});{publish}
 *due=m->eventsPending()?m->nextTimeSlot():~0ULL;return ++m->sequence;
}}
'''
    sv='module cnn_production_frame('+header+');\ntimeunit 1ps;timeprecision 1ps;\n'
    sv+='''import "DPI-C" function chandle production_create();
import "DPI-C" function void production_final(chandle h);
'''
    for kind,width in [('combo',ni),('seq',nc)]:
        sv+=f'import "DPI-C" function longint unsigned production_{kind}(chandle h,longint unsigned t,input bit [{width-1}:0] data,output bit [{no-1}:0] result,output longint unsigned due);\n'
    sv+=f'''
chandle h;
wire [{ni-1}:0] inputs={{{','.join(n for n,w in reversed(normal))}}};
wire [{nc-1}:0] clocks={{{','.join(n for n,w in reversed(clocks))}}};
bit [{no-1}:0] combo_out,seq_out,temp;
longint unsigned combo_num=0,seq_num=0,combo_due,seq_due,due,scheduled=64'hffffffffffffffff,wakeup=64'hffffffffffffffff;
reg tick=0;
wire compute_begin,compute_end;wire[14:0] debug_program_index;
assign {{{','.join(n for n,w in reversed(outputs+taps))}}}=(seq_num>combo_num)?seq_out:combo_out;
initial h=production_create();
always_comb combo_num=production_combo(h,$time,inputs,combo_out,combo_due);
always @(clocks or tick)begin
 seq_num<=production_seq(h,$time,clocks,temp,seq_due);seq_out<=temp;
end
always @(combo_num or seq_num)begin
 due=(seq_num>combo_num)?seq_due:combo_due;
 if(due!=64'hffffffffffffffff && due!=scheduled)begin
  if(due<$time)$fatal(1,"Production native time reversed");
  scheduled=due;wakeup<=#(due-$time)due;
 end
end
always @(wakeup)if(wakeup==$time)tick=~tick;
// Two-state execution is admitted only at known control/valid inputs. Payload
// checks remain handshake-qualified, so idle bus X is not mistaken for data.
always @(posedge clk)if(!reset)begin
 if($isunknown({{reset,cancel,recover,begin_valid,load_valid,verify_valid,diag_valid,start_valid,rvalid,bvalid,arready,awready,wready}}))$fatal(1,"Production DPI unknown control");
 if(rvalid && $isunknown({{rdata,rresp,rlast}}))$fatal(1,"Production DPI unknown R payload");
 if(bvalid && $isunknown(bresp))$fatal(1,"Production DPI unknown B payload");
 if(load_valid && $isunknown({{load_index,load_data,load_last}}))$fatal(1,"Production DPI unknown load payload");
end
final production_final(h);
endmodule
'''
    return cpp,sv,ports,dict(input_bits=ni,clock_bits=nc,output_bits=no)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--board',type=Path,required=True);p.add_argument('--source-list',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();board=a.board.resolve()
    assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir();src=root/'source';src.mkdir();obj=root/'obj'
    bm=json.loads((board/'inputs.json').read_text());byname={Path(r['snapshot']).name:r for r in bm['inputs']}
    prior=json.loads(a.source_list.read_text());records=[]
    for old in prior['inputs']:
        name=Path(old['snapshot']).name;row=byname[name];f=Path(row['snapshot']);assert sha(f)==row['sha256']
        target=src/name;shutil.copy2(f,target)
        if name=='cnn_production_frame.sv':
            text=target.read_text()
            old_decl='wire merge_busy,compute_begin,compute_end,dma_end;wire [14:0] debug_program_index;'
            assert text.count(old_decl)==1
            text=text.replace(old_decl,'wire merge_busy,dma_end;wire compute_begin /*verilator public_flat_rd*/;wire compute_end /*verilator public_flat_rd*/;wire [14:0] debug_program_index /*verilator public_flat_rd*/;')
            target.write_text(text)
        records.append(dict(snapshot=str(target),source=str(f),sha256=sha(target),board_source_sha256=sha(f)))
    for f in board.rglob('*.vh'):
        target=src/f.name
        if not target.exists():shutil.copy2(f,target)
    cpp,sv,ports,info=generate((src/'cnn_production_frame.sv').read_text())
    (root/'production.cpp').write_text(cpp);(root/'cnn_production_frame_dpi.sv').write_text(sv)
    env=os.environ.copy();env.update(PATH=str(BIN)+';E:/GIT/Git/usr/bin;'+env['PATH'],VERILATOR_ROOT=VR.as_posix(),MSYS2_ARG_CONV_EXCL='*')
    run([BIN/'verilator_bin.exe','--cc','--timing','--sched-zero-delay','--top-module','cnn_production_frame',
         '-I'+src.as_posix(),'--timescale-override','1ns/1ps','--threads','1','--output-split','15000',
         '--output-split-cfuncs','500','-Wno-fatal','--Mdir',obj,'-CFLAGS','-DVL_TIME_CONTEXT -fPIC',
         *[r['snapshot'] for r in records]],root,'verilate',env,120)
    log=(root/'verilate.log').read_text(errors='replace')
    assert not re.search(r'%Warning-(UNOPTFLAT|LATCH|IMPLICIT|SELRANGE|STMTDLY)',log)
    make=[BIN/'mingw32-make.exe','-f','Vcnn_production_frame.mk','-j4','Vcnn_production_frame__ALL.a',
          'verilated.o','verilated_threads.o','verilated_timing.o','CXX=g++.exe','AR=ar.exe',
          'PERL=E:/GIT/Git/usr/bin/perl.exe','SHELL=E:/GIT/Git/usr/bin/sh.exe','VERILATOR_ROOT='+VR.as_posix(),
          'OPT_FAST=-O2','OPT_GLOBAL=-O2','CXXFLAGS=-std=gnu++20 -DVL_TIME_CONTEXT']
    run(make,obj,'build',env,360)
    run([BIN/'g++.exe','-O2','-std=c++20','-DVL_TIME_CONTEXT','-I'+obj.as_posix(),'-I'+(VR/'include').as_posix(),
         '-I'+(VR/'include/vltstd').as_posix(),root/'production.cpp',obj/'Vcnn_production_frame__ALL.a',
         obj/'verilated.o',obj/'verilated_threads.o',obj/'verilated_timing.o','-pthread','-latomic','-shared','-o',root/'production.dll'],root,'link',env,90)
    manifest=dict(inputs=records,board_inputs_sha256=sha(board/'inputs.json'),ports=ports,packing=info,
        wrapper_sha256=sha(root/'cnn_production_frame_dpi.sv'),bridge_sha256=sha(root/'production.cpp'),
        dll_sha256=sha(root/'production.dll'),lookahead=False,hardware_changed=False)
    (root/'inputs.json').write_text(json.dumps(manifest,indent=2))
    result=dict(status='PRODUCTION_CNN_DPI_BUILT_NOT_YET_VALIDATED',inputs_sha256=sha(root/'inputs.json'),packing=info)
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
