"""Isolated C32 classifier output on the new six-bank row-pair interface.

Derive from a hashed snapshot; never edit the hardware workflow's source.
Keep all existing RGB/C32 transport tests and add actual region decoding.
"""
import argparse,os,re,shutil,subprocess,time
from pathlib import Path
import numpy as np
from .region_detail_target import ROOT,RUN
from .runtime import file_hash,write_json
from .train_luma_regions import LEVELS
from .prepare_region_maps import project


EXTRA=r'''
 reg region;reg [10:0] region_retired;reg [1:0] packed_pixels;
 reg [47:0] packed_rgb;reg packed_valid;
 wire decoder_ready,decoder_valid,decoder_error;wire [23:0] decoded_rgb;wire [35:0] decoded_tag;
 wire decode_valid=region && active && !aborting && feature_valid[region_retired[2:0]];
 wire decoder_take=decode_valid && decoder_ready;
 wire accept_rgb=decoder_valid && !packed_valid && active && !aborting;
 wire [6:0] compact_count;
 wire packed_append=region && packed_valid && active && !aborting && compact_count<=13;
 region_luma_decode region_decoder(.clk(clk),.reset(reset),.cancel(aborting || (cmd_valid && cmd_ready)),
  .in_valid(decode_valid),.in_ready(decoder_ready),.in_data(feature_word[region_retired[2:0]]),
  .in_group(region_retired[1:0]),.in_tag({27'd0,region_retired[10:2]}),
  .out_valid(decoder_valid),.out_ready(!packed_valid && active && !aborting),.out_rgb(decoded_rgb),.out_tag(decoded_tag),.error(decoder_error));
 always @(posedge clk)begin
  if(reset || (cmd_valid && cmd_ready))begin
   region<=!reset && cmd_region;region_retired<=0;packed_pixels<=0;packed_valid<=0;packed_rgb<=0;
  end else if(active)begin
   if(aborting)begin packed_pixels<=0;packed_valid<=0;end
   else begin
    if(decoder_take)region_retired<=region_retired+11'd1;
    if(packed_append)begin packed_valid<=0;packed_pixels<=0;packed_rgb<=0;end
    if(accept_rgb)begin
     if(packed_pixels==0)packed_rgb[23:0]<=decoded_rgb^24'h808080;
     else packed_rgb[47:24]<=decoded_rgb^24'h808080;
     packed_pixels<=packed_pixels+2'd1;
     if(packed_pixels==1 || decoded_tag=={27'd0,columns}-36'd1)packed_valid<=1;
    end
   end
  end
 end
'''


def derivative(s):
    def replace(old,new):
        nonlocal s
        assert s.count(old)==1,(old,s.count(old));s=s.replace(old,new)
    replace('input wire cmd_valid,output wire cmd_ready,input wire cmd_store,cmd_rgb,','input wire cmd_valid,output wire cmd_ready,input wire cmd_store,cmd_rgb,cmd_region,')
    replace(' wire [6:0] rgb_unit=',EXTRA+'\n wire [6:0] rgb_unit=')
    replace("wire [6:0] compact_count={1'b0,rgb_count}-drop;","assign compact_count={1'b0,rgb_count}-drop;")
    replace('wire have_write=rgb ?', 'wire have_write=(rgb || region) ?')
    replace('wire [127:0] candidate_wdata=rgb ?', 'wire [127:0] candidate_wdata=(rgb || region) ?')
    replace('wire append_rgb=rgb && !aborting && (store ? arena_response:read_fire);','wire append_rgb=!aborting && (region ? packed_append:(rgb && (store ? arena_response:read_fire)));')
    replace("wire [6:0] append_size=store ?", "wire [6:0] append_size=region ? (packed_pixels==2 ? 7'd3:7'd2):store ?")
    replace("wire [127:0] append_data=store ?", "wire [127:0] append_data=region ? {80'd0,packed_rgb}:store ?")
    replace('if(store ? (write_fire && beat_slots[s]):(arena_fire && request_slots[s]))',"if(store ? (region ? (decoder_take && region_retired[2:0]==3'(s)):(write_fire && beat_slots[s])):(arena_fire && request_slots[s]))")
    replace('if(cancel || (arena_response && sw[bank]==store) || (rgb && next_count>16))', 'if(cancel || (region && decoder_error) || (arena_response && sw[bank]==store) || ((rgb || region) && next_count>16))')
    replace('if(rgb && !aborting)begin','if((rgb || region) && !aborting)begin')
    replace("size=cmd_rgb ? int'(cmd_columns)*3:int'(cmd_columns)*32;","size=(cmd_rgb || cmd_region) ? int'(cmd_columns)*3:int'(cmd_columns)*32;")
    replace('if(cmd_address[1:0]!=0 || (!cmd_rgb && cmd_address[3:0]!=0) || cmd_phase ||','if((cmd_region && (!cmd_store || cmd_rgb)) || cmd_address[1:0]!=0 || (!cmd_rgb && !cmd_region && cmd_address[3:0]!=0) || cmd_phase ||')
    return s


TEST=r'''
  $readmemh("region_golden.hex",region_golden);
  for(integer n=0;n<8;n=n+1)for(integer bank_id=0;bank_id<6;bank_id=bank_id+1)for(integer prefix_id=0;prefix_id<4;prefix_id=prefix_id+1)begin
   case(n)0:cols=1;1:cols=3;2:cols=4;3:cols=7;4:cols=37;5:cols=64;6:cols=255;7:cols=511;endcase
   test_bank=bank_id;test_base=cols==511 ? 0:16;random_stall=region_case%2;source_address=4096;destination_address=32752+prefix_id*4;
   for(integer j=0;j<cols*32;j=j+1)memory[source_address+j]=region_case%17==0 ? 8'd0:8'(j*37+region_case*19);
   for(integer j=32736;j<35000;j=j+1)memory[j]=8'h5a;
   cmd_region=0;execute(0,source_address,cols,4,0,0);
   cmd_region=1;execute(1,destination_address,cols,4,0,0);
   for(integer j=32736;j<destination_address+cols*3+16;j=j+1)begin
    if(memory[j]!==((j>=destination_address && j<destination_address+cols*3) ? region_golden[region_case*1533+j-destination_address]:8'h5a))
     $fatal(1,"REGION_PAIR_DMA pixel/guard case=%0d address=%0d got=%h",region_case,j,memory[j]);checks=checks+1;
   end
   region_case=region_case+1;
  end
  execute(0,4096,7,4,0,1);execute(1,32768,7,1,1,1);
  test_bank=5;test_base=0;random_stall=0;
  for(integer scenario=0;scenario<4;scenario=scenario+1)begin
   @(negedge clk);pause_aw=scenario==0;pause_w=scenario==1;pause_b=scenario==2;hold_arena=scenario==3;
   cmd_store=1;cmd_region=1;cmd_rgb=0;cmd_groups=4;cmd_columns=64;cmd_bank=5;cmd_base=0;cmd_address=32764;
   cmd_valid=1;do @(posedge clk);while(!cmd_ready);@(negedge clk);cmd_valid=0;
   case(scenario)0:wait(awvalid && !awready);1:wait(wvalid && !wready);2:wait(dut.state==5);3:wait(|av);endcase
   @(posedge clk);@(negedge clk);cancel=1;repeat(4)@(negedge clk);
   if(av || dut.pending || done_valid)$fatal(1,"REGION_PAIR_CANCEL ownership scenario=%0d",scenario);
   if(scenario==0 && !awvalid)$fatal(1,"REGION_PAIR_CANCEL offered AW lost");
   if(scenario==1 && !wvalid)$fatal(1,"REGION_PAIR_CANCEL held W lost");
   pause_aw=0;pause_w=0;pause_b=0;hold_arena=0;
   begin integer timeout_count;timeout_count=0;
    while(!done_valid)begin @(negedge clk);timeout_count=timeout_count+1;if(timeout_count>10000)$fatal(1,"REGION_PAIR_CANCEL drain");end
   end
   if(!error || dut.pending || rleft || wleft || bvalid || bd)$fatal(1,"REGION_PAIR_CANCEL incomplete drain");
   cancel=0;done_ready=1;@(negedge clk);done_ready=0;
   execute(1,32764,37,4,0,0);
   for(integer j=0;j<111;j=j+1)if(memory[32764+j]!==region_golden[191*1533+j])$fatal(1,"REGION_PAIR_CANCEL recovery bytes");
   $display("REGION_PAIR_CANCEL_PASS scenario=%0d",scenario);
  end
  bad_write=1;execute(1,32764,37,4,0,1);bad_write=0;execute(1,32764,37,4,0,0);
  @(negedge clk);pause_w=1;cmd_valid=1;do @(posedge clk);while(!cmd_ready);@(negedge clk);cmd_valid=0;
  wait(wvalid && !wready);@(negedge clk);reset=1;rvalid=0;bvalid=0;rleft=0;wleft=0;bd=0;held_ar=0;held_aw=0;held_w=0;held_memory=0;
  repeat(3)@(negedge clk);pause_w=0;reset=0;execute(1,32764,37,4,0,0);
  for(integer j=0;j<111;j=j+1)if(memory[32764+j]!==region_golden[191*1533+j])$fatal(1,"REGION_PAIR_RESET recovery bytes");
  $display("PASS REGION_PAIR_DMA cases=%0d byte_checks=%0d cancel=4 write_error=1 reset=1",region_case,checks);
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',default='rtl-region-row-pair-dma-v1');a=p.parse_args()
    out=ROOT/'runs'/RUN/a.name;out.mkdir(exist_ok=False);hw=ROOT.parent/'hw/efinity/prototypes/frame_execution/variants/row_pair_arena'
    files=[];records=[]
    for src in [hw/n for n in ('row_pair_arena6.sv','row_pair_aux_router.sv','row_pair_dma.sv','tb_row_pair_dma.sv')]+[Path(__file__).with_name('region_luma_decode.sv'),Path('E:/Efinity/2025.2/sim_models/verilog/efx_dpram10.v')]:
        if not src.exists() and src.name=='efx_dpram10.v':
            from .validate_region_native import NATIVE
            src=NATIVE/'frozen/efx_dpram10.v'
        dst=out/src.name;shutil.copy2(src,dst);files.append(dst);records.append(dict(source=str(src),source_sha256=file_hash(src),snapshot=str(dst)))
    dma=out/'row_pair_dma.sv';dma.write_text(derivative(dma.read_text()))
    tb=out/'tb_row_pair_dma.sv';s=tb.read_text();s=s.replace('module tb_row_pair_dma;',"module tb_row_pair_dma;\n reg cmd_region=0;reg [7:0] region_golden[0:294335];integer region_case=0;")
    mark='  $display("PASS ROW_PAIR_DMA';assert s.count(mark)==1;s=s.replace(mark,TEST+mark);tb.write_text(s)
    golden=[]
    for ci in range(192):
        n=[1,3,4,7,37,64,255,511][ci//24];q=(np.arange(n*32).reshape(n,32)*37+ci*19)%256
        if ci%17==0:q*=0
        q=np.where(q>=128,q-256,q);y=np.asarray(LEVELS[0])[q[:,:16].argmax(1)];co=np.asarray(LEVELS[1])[q[:,16:24].argmax(1)];cg=np.asarray(LEVELS[2])[q[:,24:].argmax(1)]
        rgb=np.stack((y+co-cg,y+cg,y-co-cg),1).clip(0,255)^128;golden.extend(rgb.reshape(-1).tolist()+[0]*(1533-n*3))
    (out/'region_golden.hex').write_text(''.join(f'{v:02x}\n' for v in golden))
    for r in records:r['snapshot_sha256']=file_hash(r['snapshot'])
    write_json(out/'inputs.json',dict(records=records,script_sha256=file_hash(__file__),production_modified=False))
    exe=Path('E:/quatra/questa_fse/win64');env=os.environ.copy();env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
    def run(args,label,timeout=120):
        start=time.perf_counter()
        with (out/label).open('w',encoding='utf-8') as f:r=subprocess.run(list(map(str,args)),cwd=out,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
        assert r.returncode==0,(label,r.returncode);return time.perf_counter()-start
    run([exe/'vlib.exe','work'],'vlib.log');run([exe/'vlog.exe','-sv','-timescale','1ns/1ps','+define+ROUTED_AUX',*files],'compile.log')
    assert 'Errors: 0, Warnings: 0' in (out/'compile.log').read_text()
    result=[]
    for seed in [71,391]:
        sec=run([exe/'vsim.exe','-c','work.tb_row_pair_dma',f'+SEED={seed}','-do','onerror {quit -code 1}; run -all; quit -code 0'],f'simulation_{seed}.log',600)
        body=(out/f'simulation_{seed}.log').read_text();assert 'PASS REGION_PAIR_DMA cases=192' in body and 'Errors: 0, Warnings: 0' in body
        result.append(dict(seed=seed,seconds=sec,log_sha256=file_hash(out/f'simulation_{seed}.log'),lines=re.findall(r'^# ((?:PASS|REGION_PAIR_CANCEL_PASS).*)$',body,re.M)));print(result[-1],flush=True)
    write_json(out/'result.json',dict(status='REGION_ROW_PAIR_OFFICIAL_RAM_ROUTED_AUX_PASS',runs=result,inputs_sha256=file_hash(out/'inputs.json'),cnn=False,board=False,fit=False,timing=False))
    project(ROOT/'runs'/RUN/(a.name+'-map'),'row_pair_dma',[dma,out/'region_luma_decode.sv'])
    reference=out/'reference';reference.mkdir();reference_dma=reference/'row_pair_dma.sv';shutil.copy2(hw/'row_pair_dma.sv',reference_dma)
    project(ROOT/'runs'/RUN/(a.name+'-reference-map'),'row_pair_dma',[reference_dma])


if __name__=='__main__':main()
