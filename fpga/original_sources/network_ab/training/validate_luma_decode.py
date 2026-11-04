"""Build and test the actual categorical RGB reducer with adversarial streams."""
import json,shutil
from pathlib import Path
import numpy as np
from .region_detail_target import ROOT,RUN
from .runtime import file_hash,write_json
from .build_region_rtl import build,run
from .prepare_region_maps import project
from .train_luma_regions import LEVELS

TB='''module tb_luma_decode;
 reg clk=0;always #5 clk=~clk;
 reg reset=1,cancel=0,iv=0,orr=0;wire ir,ov,error;
 reg [63:0] data;reg [1:0] grp;reg [35:0] tag;wire [23:0] rgb;wire [35:0] otag;
 region_luma_decode dut(.clk(clk),.reset(reset),.cancel(cancel),.in_valid(iv),.in_ready(ir),.in_data(data),.in_group(grp),.in_tag(tag),.out_valid(ov),.out_ready(orr),.out_rgb(rgb),.out_tag(otag),.error(error));
 reg [63:0] words[0:NUM*4-1];reg [23:0] golden[0:NUM-1];integer received=0,held_checks=0,cycles=0;reg scoreboard=1,random_stall=1;
 reg [31:0] rng=32'h8297413;reg held=0;reg [59:0] saved;
 always @(negedge clk)begin rng=rng^(rng<<13);rng=rng^(rng>>17);rng=rng^(rng<<5);orr=!random_stall || rng[0] || rng[1];end
 always @(posedge clk)if(!reset && !cancel)begin
  cycles=cycles+1;if(cycles>NUM*30+1000)$fatal(1,"timeout");
  if(held && (!ov || {otag,rgb}!==saved))$fatal(1,"output changed under backpressure");
  if(ov && !orr)begin held_checks=held_checks+1;saved={otag,rgb};end
  held=ov && !orr;
  if(ov && orr && scoreboard)begin
   if(received>=NUM || rgb!==golden[received] || otag!==36'habc000000+36'(received))$fatal(1,"numeric or tag mismatch i=%0d got=%h expected=%h",received,rgb,golden[received]);
   received=received+1;
  end
 end else held=0;
 task send(input [63:0] d,input [1:0] g,input [35:0] t);
  begin @(negedge clk);iv=1;data=d;grp=g;tag=t;@(posedge clk);while(!ir)@(posedge clk);@(negedge clk);iv=0;end
 endtask
 task clear(input bit hard);
  begin @(negedge clk);if(hard)reset=1;else cancel=1;@(negedge clk);reset=0;cancel=0;if(ov || error)$fatal(1,"clear failed");end
 endtask
 initial begin
  $readmemh("words.hex",words);$readmemh("golden.hex",golden);repeat(3)@(negedge clk);reset=0;
  for(integer i=0;i<NUM;i=i+1)begin
   for(integer g=0;g<4;g=g+1)send(words[i*4+g],2'(g),36'habc000000+36'(i));
   if(i%17==0)repeat(3)@(negedge clk);
  end
  while(received<NUM)@(negedge clk);scoreboard=0;
  send(0,2,0);if(!error || ir)$fatal(1,"group order not rejected");clear(0);
  send(0,0,11);send(0,1,12);if(!error)$fatal(1,"tag mismatch not rejected");clear(1);
  send(0,0,0);send(0,1,0);clear(0);for(integer g=0;g<4;g=g+1)send(0,2'(g),123);
  while(!ov)@(negedge clk);if(rgb!==24'h00_00_00 || otag!==123)$fatal(1,"equal negative tie/recovery");clear(1);
  if(held_checks==0)$fatal(1,"no backpressure coverage");
  $display("PASS LUMA_DECODE pixels=%0d held=%0d bad_group=1 bad_tag=1 reset_cancel=1 cycles=%0d",received,held_checks,cycles);$finish;
 end
endmodule
'''

def main():
    root=ROOT/'runs'/RUN/'rtl-luma-decode';root.mkdir(exist_ok=False);src=Path(__file__).with_name('region_luma_decode.sv');shutil.copy2(src,root/src.name)
    rng=np.random.default_rng(2026100426);random=rng.integers(-128,128,(4096,32),dtype=np.int16)
    # All Y/Co/Cg combinations including clipping; independent signed maxima.
    combinations=np.full((1024,32),-128,np.int16)
    for i in range(1024):combinations[i,[i//64,16+(i//8)%8,24+i%8]]=127
    # Exhaustive signed-pair comparison across the two Y groups.
    pair=np.full((65536,32),-128,np.int16);pair[:,0]=np.repeat(np.arange(-128,128),256);pair[:,15]=np.tile(np.arange(-128,128),256)
    q=np.concatenate((random,combinations,pair));yi=q[:,:16].argmax(1);coi=q[:,16:24].argmax(1);cgi=q[:,24:].argmax(1)
    y=np.array(LEVELS[0])[yi];co=np.array(LEVELS[1])[coi];cg=np.array(LEVELS[2])[cgi]
    rgb=np.stack((y+co-cg,y+cg,y-co-cg),1).clip(0,255)
    words=q.astype(np.int8).view(np.uint8).reshape(-1,8)
    (root/'words.hex').write_text(''.join(f'{int.from_bytes(row.tobytes(),"little"):016x}\n' for row in words))
    (root/'golden.hex').write_text(''.join(f'{int(r)|(int(g)<<8)|(int(b)<<16):06x}\n' for r,g,b in rgb))
    # Equal logits select first bins, reconstruct (0,-96,-64)->(0,0,160).
    tb=root/'tb_luma_decode.sv';tb.write_text(TB.replace('NUM',str(len(q))).replace("24'h00_00_00","24'ha0_00_00"))
    write_json(root/'inputs.json',dict(rtl_sha256=file_hash(src),testbench_sha256=file_hash(tb),vectors_sha256=file_hash(root/'words.hex'),
        golden_sha256=file_hash(root/'golden.hex'),pixels=len(q),all_yco_cg_combinations=1024,exhaustive_signed_pairs=65536))
    build(root,[root/src.name,tb],'tb_luma_decode');binary=root/'obj/Vtb_luma_decode.exe';seconds=run([binary],root,root/'simulation.log',60)
    body=(root/'simulation.log').read_text();assert 'PASS LUMA_DECODE' in body
    write_json(root/'result.json',dict(status='ISOLATED_REDUCER_PASS',seconds=seconds,line=body.splitlines()[0],board_integration=False))
    project(ROOT/'runs'/RUN/'map-luma-decode','region_luma_decode',[src]);print(body,flush=True)

if __name__=='__main__':main()
