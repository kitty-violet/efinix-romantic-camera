"""C3 component decoder and six-bank RGB DMA, isolated real-RAM tests."""
import argparse,os,re,shutil,subprocess,time
from pathlib import Path
import numpy as np
from .validate_region_row_pair_dma import TEST
from .train_luma_regions import ROOT,RUN,LEVELS
from .runtime import file_hash,write_json
from .prepare_region_maps import project
from .build_region_rtl import build,run as run_verilator
from .validate_region_native import NATIVE


def derivative(s):
    def replace(old,new):
        nonlocal s
        assert s.count(old)==1,(old,s.count(old));s=s.replace(old,new)
    replace('input wire cmd_valid,output wire cmd_ready,input wire cmd_store,cmd_rgb,','input wire cmd_valid,output wire cmd_ready,input wire cmd_store,cmd_rgb,cmd_region,')
    extra=''' reg region;
 wire [23:0] low_rgb,high_rgb;
 component_pixel_decode low_decode(.value(arena_data[23:0]),.rgb(low_rgb));
 component_pixel_decode high_decode(.value(arena_data[87:64]),.rgb(high_rgb));
 always @(posedge clk)if(reset || (cmd_valid && cmd_ready))region<=!reset && cmd_region;
'''
    replace(' wire second=',extra+' wire second=')
    replace("wire [127:0] append_data=store ? {80'd0,arena_data[87:64],arena_data[23:0]}:read_payload;","wire [127:0] append_data=store ? (region ? {80'd0,high_rgb^24'h808080,low_rgb^24'h808080}:{80'd0,arena_data[87:64],arena_data[23:0]}):read_payload;")
    replace('if(cmd_address[1:0]!=0 ||','if((cmd_region && (!cmd_store || !cmd_rgb)) || cmd_address[1:0]!=0 ||')
    return s


def exhaustive(out,decoder):
    dst=out/'exhaustive';dst.mkdir();levels=[]
    for j,lev in enumerate(LEVELS):
        raw=np.arange(256);value=np.where(raw>=128,raw-256,raw)+(128 if j==0 else 0);lev=np.asarray(lev)
        q=lev[np.abs(value[:,None]-lev).argmin(1)];levels.append(q)
        (dst/f'component{j}.hex').write_text(''.join(f'{int(v)&1023:03x}\n' for v in q))
    tb=dst/'tb_component_decode.sv';tb.write_text('''module tb_component_decode;
 reg [23:0] value;wire [23:0] rgb;reg signed [9:0] gy[0:255],gco[0:255],gcg[0:255];
 component_pixel_decode dut(.*);
 function automatic [7:0] clip(input integer x);if(x<0)clip=0;else if(x>255)clip=255;else clip=8'(x);endfunction
 initial begin
  $readmemh("component0.hex",gy);$readmemh("component1.hex",gco);$readmemh("component2.hex",gcg);
  for(integer i=0;i<16777216;i=i+1)begin
   value=24'(i);#1;
   if(rgb!=={clip(int'(gy[value[7:0]])-int'(gco[value[15:8]])-int'(gcg[value[23:16]])),clip(int'(gy[value[7:0]])+int'(gcg[value[23:16]])),clip(int'(gy[value[7:0]])+int'(gco[value[15:8]])-int'(gcg[value[23:16]]))})$fatal(1,"COMPONENT mismatch %h",value);
  end
  $display("PASS COMPONENT_DECODE_EXHAUSTIVE 16777216");$finish;
 end
endmodule
''')
    frozen=dst/decoder.name;shutil.copy2(decoder,frozen);build(dst,[frozen,tb],'tb_component_decode')
    seconds=run_verilator([dst/'obj/Vtb_component_decode.exe'],dst,dst/'simulation.log',120)
    assert 'PASS COMPONENT_DECODE_EXHAUSTIVE 16777216' in (dst/'simulation.log').read_text();return dict(combinations=16777216,seconds=seconds,log_sha256=file_hash(dst/'simulation.log'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',default='rtl-component-row-pair-dma-v1');a=p.parse_args()
    out=ROOT/'runs'/RUN/a.name;out.mkdir(exist_ok=False);hw=ROOT.parent/'hw/efinity/prototypes/frame_execution/variants/row_pair_arena';records=[];files=[]
    for src in [hw/n for n in ('row_pair_arena6.sv','row_pair_aux_router.sv','row_pair_dma.sv','tb_row_pair_dma.sv')]+[Path(__file__).with_name('component_pixel_decode.sv'),NATIVE/'frozen/efx_dpram10.v']:
        dst=out/src.name;shutil.copy2(src,dst);records.append(dict(source=str(src),source_sha256=file_hash(src),snapshot=str(dst)));files.append(dst)
    reference=out/'reference';reference.mkdir();shutil.copy2(out/'row_pair_dma.sv',reference/'row_pair_dma.sv')
    dma=out/'row_pair_dma.sv';dma.write_text(derivative(dma.read_text()))
    test=TEST.replace('REGION_PAIR','COMPONENT_PAIR').replace('cols*32','cols*3')
    test=test.replace('execute(0,source_address,cols,4,0,0)','execute(0,source_address,cols,1,1,0)').replace('execute(1,destination_address,cols,4,0,0)','execute(1,destination_address,cols,1,1,0)')
    test=test.replace('execute(1,32768,7,1,1,1)','execute(1,32768,7,4,0,1)')
    test=test.replace('cmd_rgb=0;cmd_groups=4','cmd_rgb=1;cmd_groups=1').replace('37,4,0,','37,1,1,')
    tb=out/'tb_row_pair_dma.sv';s=tb.read_text().replace('module tb_row_pair_dma;',"module tb_row_pair_dma;\n reg cmd_region=0;reg [7:0] region_golden[0:294335];integer region_case=0;")
    mark='  $display("PASS ROW_PAIR_DMA';assert s.count(mark)==1;tb.write_text(s.replace(mark,test+mark))
    golden=[]
    for ci in range(192):
        n=[1,3,4,7,37,64,255,511][ci//24];q=(np.arange(n*3).reshape(n,3)*37+ci*19)%256
        if ci%17==0:q*=0
        q=np.where(q>=128,q-256,q);q[:,0]+=128;values=[]
        for j,lev in enumerate(LEVELS):lev=np.asarray(lev);values.append(lev[np.abs(q[:,j,None]-lev).argmin(1)])
        y,co,cg=values;rgb=np.stack((y+co-cg,y+cg,y-co-cg),1).clip(0,255)^128;golden.extend(rgb.reshape(-1).tolist()+[0]*(1533-n*3))
    (out/'region_golden.hex').write_text(''.join(f'{v:02x}\n' for v in golden))
    for r in records:r['snapshot_sha256']=file_hash(r['snapshot'])
    write_json(out/'inputs.json',dict(records=records,script_sha256=file_hash(__file__),production_modified=False))
    arithmetic=exhaustive(out,out/'component_pixel_decode.sv');exe=Path('E:/quatra/questa_fse/win64');env=os.environ.copy();env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
    def run(args,label,timeout=120):
        start=time.perf_counter()
        with (out/label).open('w',encoding='utf-8') as f:r=subprocess.run(list(map(str,args)),cwd=out,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
        assert r.returncode==0,(label,r.returncode);return time.perf_counter()-start
    run([exe/'vlib.exe','work'],'vlib.log');run([exe/'vlog.exe','-sv','-timescale','1ns/1ps','+define+ROUTED_AUX',*files],'compile.log');assert 'Errors: 0, Warnings: 0' in (out/'compile.log').read_text()
    runs=[]
    for seed in [71,391]:
        sec=run([exe/'vsim.exe','-c','work.tb_row_pair_dma',f'+SEED={seed}','-do','onerror {quit -code 1}; run -all; quit -code 0'],f'simulation_{seed}.log',600)
        body=(out/f'simulation_{seed}.log').read_text();assert 'PASS COMPONENT_PAIR_DMA cases=192' in body and 'Errors: 0, Warnings: 0' in body
        runs.append(dict(seed=seed,seconds=sec,log_sha256=file_hash(out/f'simulation_{seed}.log'),lines=re.findall(r'^# ((?:PASS|COMPONENT_PAIR_CANCEL_PASS).*)$',body,re.M)));print(runs[-1],flush=True)
    write_json(out/'result.json',dict(status='COMPONENT_ROW_PAIR_OFFICIAL_RAM_ROUTED_AUX_PASS',arithmetic=arithmetic,runs=runs,inputs_sha256=file_hash(out/'inputs.json'),cnn=False,board=False,fit=False,timing=False))
    project(ROOT/'runs'/RUN/(a.name+'-map'),'row_pair_dma',[dma,out/'component_pixel_decode.sv']);project(ROOT/'runs'/RUN/(a.name+'-reference-map'),'row_pair_dma',[reference/'row_pair_dma.sv'])


if __name__=='__main__':main()
