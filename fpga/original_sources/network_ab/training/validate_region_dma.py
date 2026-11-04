"""Reuse the existing DMA return routing for C32 -> RGB output.

Produces an isolated derived module. No production RTL is modified. Keeps the
normal RGB/C32 paths and runs their existing cancellation regression first.
"""
import argparse,shutil
from pathlib import Path
import numpy as np
from .region_detail_target import ROOT,RUN
from .runtime import file_hash,write_json
from .validate_region_native import NATIVE
from .build_region_rtl import build,run
from .prepare_region_maps import project
from .train_luma_regions import LEVELS


DECODE=r'''
 reg region;reg [8:0] columns;reg [2:0] packed_count;reg [95:0] packed_rgb;reg packed_valid;
 wire decoder_ready,decoder_valid,decoder_error;wire [23:0] decoded_rgb;wire [35:0] decoded_tag;
 wire decoder_take=region && active && !aborting && retired<issued && slot_valid[retired[2:0]] && decoder_ready;
 wire accept_rgb=decoder_valid && !packed_valid && active && !aborting;
 wire packed_append=region && packed_valid && active && !aborting && int'(compact_count)+int'(packed_count)*3<=32;
 region_luma_decode region_decoder(.clk(clk),.reset(reset),.cancel(aborting || (cmd_valid && cmd_ready)),
  .in_valid(region && active && !aborting && retired<issued && slot_valid[retired[2:0]]),.in_ready(decoder_ready),
  .in_data(slot_data[retired[2:0]]),.in_group(retired[1:0]),.in_tag({27'd0,retired[10:2]}),
  .out_valid(decoder_valid),.out_ready(!packed_valid && active && !aborting),.out_rgb(decoded_rgb),.out_tag(decoded_tag),.error(decoder_error));
 always @(posedge clk)begin
  if(reset || (cmd_valid && cmd_ready))begin
   region<=!reset && cmd_region;columns<=cmd_columns;packed_count<=0;packed_valid<=0;packed_rgb<=0;
  end else if(active)begin
   if(aborting)begin packed_count<=0;packed_valid<=0;end
   else begin
    if(packed_append)begin packed_valid<=0;packed_count<=0;packed_rgb<=0;end
    if(accept_rgb)begin
     packed_rgb[packed_count*24+:24]<=decoded_rgb^24'h808080;packed_count<=packed_count+1'b1;
     if(packed_count==3 || decoded_tag=={27'd0,columns}-36'd1)packed_valid<=1;
    end
   end
  end
 end
'''


def derivative(source):
    s=source.read_text()
    def replace(old,new):
        nonlocal s
        assert s.count(old)==1,(old,s.count(old));s=s.replace(old,new)
    replace('input wire cmd_valid,output wire cmd_ready,input wire cmd_store,cmd_rgb,','input wire cmd_valid,output wire cmd_ready,input wire cmd_store,cmd_rgb,cmd_region,')
    replace(' wire [3:0] unit=rgb ?',DECODE+'\n wire [3:0] unit=rgb ?')
    replace('  if(rgb && retire_count<4 && int\'(retired)+retire_count<int\'(words))retire_count=0;',
            '  if(rgb && retire_count<4 && int\'(retired)+retire_count<int\'(words))retire_count=0;\n  if(region)retire_count={2\'d0,decoder_take};')
    replace("wire [6:0] append_bytes=7'(rgb ? retire_count*3:retire_count*8);","wire [6:0] append_bytes=region ? 7'(packed_count*3):7'(rgb ? retire_count*3:retire_count*8);")
    replace('wire append_valid=store ? retire_count!=0:(read_fire && !aborting);','wire append_valid=store ? (region ? packed_append:retire_count!=0):(read_fire && !aborting);')
    replace('wire [127:0] append_payload=store ? append_words:read_payload;','wire [127:0] append_payload=store ? (region ? {32\'d0,packed_rgb}:append_words):read_payload;')
    replace('if(cancel || response_bad || next_count>32 ||','if(cancel || response_bad || (region && decoder_error) || next_count>32 ||')
    replace("size=cmd_rgb ? int'(cmd_columns)*3:nwords*8;","size=(cmd_rgb || cmd_region) ? int'(cmd_columns)*3:nwords*8;")
    replace("if(cmd_address[1:0]!=0 || (!cmd_rgb && cmd_address[3:0]!=0) ||","if((cmd_region && (!cmd_store || cmd_rgb)) || cmd_address[1:0]!=0 || (!cmd_rgb && !cmd_region && cmd_address[3:0]!=0) ||")
    return s


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);a=ap.parse_args();root=ROOT/'runs'/RUN/a.name;root.mkdir(exist_ok=False)
    hw=ROOT.parent/'hw/efinity/prototypes/frame_execution';source=hw/'variants/fixed_storage/frame_arena_row_dma.sv'
    dst=root/source.name;dst.write_text(derivative(source));files=[dst]
    for f in [Path(__file__).with_name('region_luma_decode.sv'),NATIVE/'frozen/tensor_arena12.sv',NATIVE/'frozen/efx_dpram10.v']:
        d=root/f.name;shutil.copy2(f,d);files.append(d)
    tbsource=hw/'variants/fixed_parcel_dma/tb_frame_arena_row_dma.sv';s=tbsource.read_text().replace('module tb_frame_arena_row_dma;','module tb_region_dma;\n reg cmd_region=0;reg [7:0] region_golden[0:147167];integer region_case=0;')
    # The existing 30 good transfers/21 reject/cancel cases remain unchanged.
    mark='  $display("PASS ROW_DMA_SHARED_ABORT'
    assert s.count(mark)==1
    s=s.replace(mark,'''  $readmemh("region_golden.hex",region_golden);
  for(integer n=0;n<8;n=n+1)for(integer pl=0;pl<3;pl=pl+1)for(integer prefix_id=0;prefix_id<4;prefix_id=prefix_id+1)begin
   case(n)0:cols=1;1:cols=3;2:cols=4;3:cols=7;4:cols=37;5:cols=64;6:cols=255;7:cols=511;endcase
   test_plane=pl;test_base=16;random_stall=region_case%2;source_address=4096;destination_address=32752+prefix_id*4;
   for(integer j=0;j<cols*32;j=j+1)memory[source_address+j]=8'(j*37+region_case*19);
   for(integer j=32736;j<35000;j=j+1)memory[j]=8'h5a;
   cmd_region=0;execute(0,source_address,cols,4,0,0);
   cmd_region=1;execute(1,destination_address,cols,4,0,0);
   for(integer j=32736;j<destination_address+cols*3+16;j=j+1)begin
    if(memory[j]!==((j>=destination_address && j<destination_address+cols*3) ? region_golden[region_case*1533+j-destination_address]:8'h5a))
     $fatal(1,"REGION_DMA pixel/guard case=%0d address=%0d got=%h",region_case,j,memory[j]);checks=checks+1;
   end
   region_case=region_case+1;
  end
  // Reject attempts to decode loads or mix RGB input and C32 logits.
  execute(0,4096,7,4,0,1);execute(1,32768,7,1,1,1);
  test_plane=2;test_base=16;random_stall=0;
  for(integer scenario=0;scenario<4;scenario=scenario+1)begin
   @(negedge clk);pause_aw=scenario==0;pause_w=scenario==1;pause_b=scenario==2;hold_arena=scenario==3;
   cmd_store=1;cmd_region=1;cmd_rgb=0;cmd_groups=4;cmd_columns=64;cmd_plane=2;cmd_base=16;cmd_address=32764;
   cmd_valid=1;do @(posedge clk);while(!cmd_ready);@(negedge clk);cmd_valid=0;
   case(scenario)0:wait(awvalid && !awready);1:wait(wvalid && !wready);2:wait(dut.state==5);3:wait(|av);endcase
   @(posedge clk);@(negedge clk);cancel=1;repeat(4)@(negedge clk);
   if(av || dut.pending || dut.held_lanes || done_valid)$fatal(1,"REGION_CANCEL ownership scenario=%0d",scenario);
   if(scenario==0 && !awvalid)$fatal(1,"REGION_CANCEL offered AW lost");
   if(scenario==1 && !wvalid)$fatal(1,"REGION_CANCEL held W lost");
   pause_aw=0;pause_w=0;pause_b=0;hold_arena=0;
   begin integer timeout_count;timeout_count=0;
    while(!done_valid)begin @(negedge clk);timeout_count=timeout_count+1;if(timeout_count>10000)$fatal(1,"REGION_CANCEL drain");end
   end
   if(!error || dut.pending || rleft || wleft || bvalid || bd)$fatal(1,"REGION_CANCEL incomplete drain");
   cancel=0;done_ready=1;@(negedge clk);done_ready=0;
   execute(1,32764,37,4,0,0);
   for(integer j=0;j<111;j=j+1)if(memory[32764+j]!==region_golden[95*1533+j])$fatal(1,"REGION_CANCEL recovery bytes");
   $display("REGION_CANCEL_PASS scenario=%0d",scenario);
  end
  bad_write=1;execute(1,32764,37,4,0,1);bad_write=0;execute(1,32764,37,4,0,0);
  @(negedge clk);pause_w=1;cmd_valid=1;do @(posedge clk);while(!cmd_ready);@(negedge clk);cmd_valid=0;
  wait(wvalid && !wready);@(negedge clk);reset=1;rvalid=0;bvalid=0;rleft=0;wleft=0;bd=0;held_ar=0;held_aw=0;held_w=0;held_memory=0;
  repeat(3)@(negedge clk);pause_w=0;reset=0;execute(1,32764,37,4,0,0);
  for(integer j=0;j<111;j=j+1)if(memory[32764+j]!==region_golden[95*1533+j])$fatal(1,"REGION_RESET recovery bytes");
  cmd_region=0;
  $display("PASS REGION_DMA cases=%0d byte_checks=%0d cancel=4 write_error=1 reset=1",region_case,checks);
'''+mark)
    golden=[]
    for ci in range(96):
        n=[1,3,4,7,37,64,255,511][ci//12];q=(np.arange(n*32).reshape(n,32)*37+ci*19)%256;q=np.where(q>=128,q-256,q)
        y=np.asarray(LEVELS[0])[q[:,:16].argmax(1)];co=np.asarray(LEVELS[1])[q[:,16:24].argmax(1)];cg=np.asarray(LEVELS[2])[q[:,24:].argmax(1)]
        rgb=np.stack((y+co-cg,y+cg,y-co-cg),1).clip(0,255)^128;golden.extend(rgb.reshape(-1).tolist()+[0]*(1533-n*3))
    (root/'region_golden.hex').write_text(''.join(f'{v:02x}\n' for v in golden));tb=root/'tb_region_dma.sv';tb.write_text(s);files.append(tb)
    write_json(root/'inputs.json',dict(source=str(source),source_sha256=file_hash(source),tb_source=str(tbsource),tb_sha256=file_hash(tbsource),files={str(f):file_hash(f) for f in files},production_modified=False))
    build(root,files,'tb_region_dma');seconds=run([root/'obj/Vtb_region_dma.exe'],root,root/'simulation.log',90)
    text=(root/'simulation.log').read_text();assert 'PASS REGION_DMA cases=96' in text
    write_json(root/'result.json',dict(status='REGION_DMA_BASELINE_NUMERIC_AND_RECOVERY_PASS',seconds=seconds,board=False,region_cancellation_tested=True,lines=[l for l in text.splitlines() if l.startswith(('PASS','REGION_CANCEL_PASS'))]))
    project(ROOT/'runs'/RUN/(a.name+'-map'),'frame_arena_row_dma',files[:2]);project(ROOT/'runs'/RUN/(a.name+'-reference-map'),'frame_arena_row_dma',[source])
    print(text[-1800:],flush=True)


if __name__=='__main__':main()
