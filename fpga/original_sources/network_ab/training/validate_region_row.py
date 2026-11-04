"""Actual Ti60 RAM simulation of the new C32-to-RGB row converter."""
import argparse,shutil
from pathlib import Path
import numpy as np
from .region_detail_target import ROOT,RUN
from .runtime import file_hash,write_json
from .validate_region_native import NATIVE,byte
from .build_region_rtl import build,run
from .prepare_region_maps import project
from .train_luma_regions import LEVELS

TB='''module tb_region_row;
 reg clk=0;always #5 clk=~clk;
 reg reset=1,cancel=0,mode=0,stall=0,inject=0,cv=0,dr=0;
 reg [8:0] pixels;reg [1:0] source_plane,output_plane;reg [9:0] source_base,output_base;
 wire cr,dv,busy,error;wire [31:0] cycles,reads,writes,acks;
 wire [23:0] v,ready,w,rspv,rspr,rspw,ar,sv,sw;wire [239:0] addr;wire [1535:0] data,sd;
 reg [23:0] tv=0,tw=0,req_gate=24'hffffff,rsp_gate=24'hffffff;reg [239:0] ta=0;reg [1535:0] td=0;
 wire [23:0] av=mode ? v&req_gate:tv,aw=mode ? w:tw,sr=mode ? rspr&rsp_gate:24'hffffff;
 wire [239:0] aa=mode ? addr:ta;wire [1535:0] ad=mode ? data:td;
 assign ready=ar&req_gate;assign rspv=sv&rsp_gate;assign rspw=sw^{24{inject}};
 tensor_arena12 arena(.clk(clk),.reset(reset),.cancel(cancel),.req_valid(av),.req_ready(ar),.req_write(aw),.req_address(aa),.req_data(ad),.req_tag(384'd0),.rsp_valid(sv),.rsp_ready(sr),.rsp_data(sd),.rsp_tag(),.rsp_was_write(sw));
 region_row_decode dut(.clk(clk),.reset(reset),.cancel(cancel),.cmd_valid(cv),.cmd_ready(cr),.cmd_pixels(pixels),.cmd_source_plane(source_plane),.cmd_output_plane(output_plane),.cmd_source_base(source_base),.cmd_output_base(output_base),
 .done_valid(dv),.done_ready(dr),.busy(busy),.error(error),.av(v),.ar(ready),.aw(w),.aa(addr),.ad(data),.sv(rspv),.sr(rspr),.sd(sd),.sw(rspw),.cycles(cycles),.reads(reads),.writes(writes),.acks(acks));
 reg [63:0] golden[0:NUM*256-1];reg [31:0] rng=32'h179482;integer ci=0,checks=0,held_checks=0,waits;
 reg [23:0] hold_req=0,exposed_req=0,exposed_rsp=0;reg [239:0] ha;reg [1535:0] hd;reg [23:0] hw;
 function [63:0] memword(input integer b,a,c);
  reg [63:0] x;begin for(integer l=0;l<8;l=l+1)x[l*8+:8]=8'(((b*37+a*13+l*29+c*11)&255)-128);memword=x;end
 endfunction
 always @(negedge clk)begin
  if(stall && mode)begin rng=rng^(rng<<13);rng=rng^(rng>>17);rng=rng^(rng<<5);req_gate=rng[23:0]|(rng[23:0]<<1)|exposed_req;
   rng=rng^(rng<<13);rng=rng^(rng>>17);rng=rng^(rng<<5);rsp_gate=rng[23:0]|(rng[23:0]>>1)|exposed_rsp;end
  else begin req_gate=24'hffffff;rsp_gate=24'hffffff;end
 end
 always @(posedge clk)begin
  if(mode && !reset && !cancel)begin
   for(integer p=0;p<24;p=p+1)if(hold_req[p])begin if(!v[p] || addr[p*10+:10]!==ha[p*10+:10] || data[p*64+:64]!==hd[p*64+:64] || w[p]!==hw[p])$fatal(1,"held request changed p=%0d",p);held_checks=held_checks+1;end
   hold_req=v&~ready;exposed_req=av&~ar;exposed_rsp=sv&rsp_gate&~sr;
  end else begin hold_req=0;exposed_req=0;exposed_rsp=0;end
  ha=addr;hd=data;hw=w;
 end
 task initialize;
  begin mode=0;stall=0;
   for(integer a=0;a<1024;a=a+2)begin
    tv=24'hffffff;tw=24'hffffff;
    for(integer p=0;p<24;p=p+1)begin ta[p*10+:10]=10'(a+p%2);td[p*64+:64]=memword(p/2,a+p%2,ci);end
    @(posedge clk);if(ar!==24'hffffff)$fatal(1,"init blocked");@(negedge clk);
   end
   tv=0;tw=0;@(negedge clk);
  end
 endtask
 task start;
  begin @(negedge clk);mode=1;cv=1;do @(posedge clk);while(!cr);@(negedge clk);cv=0;end
 endtask
 task finish(input bit bad);
  begin waits=0;while(!dv)begin @(negedge clk);waits=waits+1;if(waits>10000)$fatal(1,"timeout case=%0d issued=%0d received=%0d pending=%h have=%h error=%b",ci,dut.issued[0],dut.received[0],dut.pending,dut.have,error);end
   if(error!==bad)$fatal(1,"wrong error");repeat(3)@(negedge clk);if(cr)$fatal(1,"early restart");dr=1;@(negedge clk);dr=0;
  end
 endtask
 task verify;
  integer port_n,a;begin mode=0;stall=0;
   for(integer x=-4;x<int'(pixels)+4;x=x+1)begin
    if(x<0 || x>=int'(pixels))begin
     port_n=(int'(output_plane)*4+((x%4+4)%4))*2;a=int'(output_base)+(x<0 ? -1:(int'(pixels)+3)/4);
    end else begin port_n=(int'(output_plane)*4+x%4)*2;a=int'(output_base)+x/4;end
    @(negedge clk);tv=24'd1<<port_n;tw=0;ta=0;ta[port_n*10+:10]=10'(a);
    @(posedge clk);if(!ar[port_n])$fatal(1,"read not ready");@(negedge clk);tv=0;
    if(!sv[port_n] || sw[port_n])$fatal(1,"read response");
    if(x>=0 && x<int'(pixels))begin
     if(sd[port_n*64+:64]!==golden[ci*256+x])$fatal(1,"numeric case=%0d x=%0d got=%h expected=%h",ci,x,sd[port_n*64+:64],golden[ci*256+x]);
    end else if(sd[port_n*64+:64]!==memword(port_n/2,a,ci))$fatal(1,"guard overwritten");
    checks=checks+1;
   end
   @(negedge clk);
  end
 endtask
 initial begin
  $readmemh("golden.hex",golden);repeat(3)@(negedge clk);reset=0;
  for(ci=0;ci<NUM;ci=ci+1)begin
   case(ci%6)0:pixels=1;1:pixels=3;2:pixels=4;3:pixels=7;4:pixels=64;5:pixels=256;endcase
   source_plane=2'((ci/6)%3);output_plane=2'((int'(source_plane)+1+(ci/18)%2)%3);source_base=10'(ci%3*100);output_base=960;
   initialize();stall=ci>=36;start();finish(0);
   if(reads!==32'(int'(pixels)*4) || writes!==32'(pixels) || acks!==32'(pixels))$fatal(1,"accounting");
   $display("ROW case=%0d width=%0d random=%0d cycles=%0d",ci,pixels,ci>=36,cycles);verify();
  end
  source_plane=1;output_plane=1;pixels=7;start();finish(1);if(reads || writes)$fatal(1,"bad command had side effect");
  source_plane=0;output_plane=1;source_base=0;output_base=960;pixels=64;
  for(integer a=0;a<4;a=a+1)begin
   start();repeat(2+a*3)@(negedge clk);cancel=1;@(negedge clk);cancel=0;if(busy)$fatal(1,"cancel not idle");
  end
  ci=4;source_base=100;initialize();start();finish(0);verify();
  if(held_checks==0)$fatal(1,"no hold coverage");
  $display("PASS REGION_ROW cases=%0d pixels_checked=%0d holds=%0d invalid=1 shared_cancel_recovery=4",NUM,checks,held_checks);$finish;
 end
endmodule
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',default='rtl-row-decode-static');p.add_argument('--map-name',default='map-row-decode-static');a=p.parse_args()
    root=ROOT/'runs'/RUN/a.name;root.mkdir(exist_ok=False);source=Path(__file__).parent;files=[]
    for f in [source/'region_luma_decode.sv',source/'region_row_decode.sv',NATIVE/'frozen/tensor_arena12.sv',NATIVE/'frozen/efx_dpram10.v']:
        dst=root/f.name;shutil.copy2(f,dst);files.append(dst)
    rows=[];cases=[]
    for ci in range(72):
        w=[1,3,4,7,64,256][ci%6];plane=(ci//6)%3;base=ci%3*100;out=[]
        for x in range(w):
            q=[byte(plane*4+x%4,base+x//4*4+g,l,ci) for g in range(4) for l in range(8)]
            y=LEVELS[0][int(np.argmax(q[:16]))];co=LEVELS[1][int(np.argmax(q[16:24]))];cg=LEVELS[2][int(np.argmax(q[24:]))]
            rgb=[max(0,min(255,v))^128 for v in (y+co-cg,y+cg,y-co-cg)]
            out.append(sum(v<<(8*i) for i,v in enumerate(rgb)))
        rows.extend(out+[0]*(256-w));cases.append(dict(width=w,source_plane=plane,base=base))
    (root/'golden.hex').write_text(''.join(f'{r:016x}\n' for r in rows));tb=root/'tb_region_row.sv';tb.write_text(TB.replace('NUM','72'));files.append(tb)
    write_json(root/'inputs.json',dict(files={str(f):file_hash(f) for f in files},cases=cases,golden_sha256=file_hash(root/'golden.hex'),actual_official_RAM=True))
    build(root,files,'tb_region_row');seconds=run([root/'obj/Vtb_region_row.exe'],root,root/'simulation.log',60)
    body=(root/'simulation.log').read_text();assert 'PASS REGION_ROW' in body
    write_json(root/'result.json',dict(status='ACTUAL_ARENA_ROW_CONVERSION_PASS',seconds=seconds,lines=[l for l in body.splitlines() if l.startswith(('ROW ','PASS '))],board_integration=False))
    project(ROOT/'runs'/RUN/a.map_name,'region_row_decode',[source/'region_luma_decode.sv',source/'region_row_decode.sv'])
    print(body[-1500:],flush=True)

if __name__=='__main__':main()
