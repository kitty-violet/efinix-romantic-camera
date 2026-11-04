"""Bounded real fetch/issuer/header/core/DMA coupling on trained weights.

Uses the full checked descriptor image, but stops after eight actual output
rows in the first stripe. No testbench intermediate activation injection.
This is explicitly not a full-frame, PHY, board-fit or timing sign-off.
"""
import argparse,json,shutil,sys
from pathlib import Path
import numpy as np
from PIL import Image
from .region_detail_target import ROOT,RUN
from .runtime import file_hash,write_json
from .build_region_rtl import build,run
from .probe_boundary_budget import HW


BODY=r'''
 reg [63:0] coefficients[0:2559];
 reg [511:0] compute_image[0:34199];reg [127:0] dma_image[0:28799];
 reg [7:0] source_image[0:2764799],golden[0:1535];reg [1535:0] seen=0;
 integer output_bytes=0,output_rows=0,ticks=0,reads_seen=0,held_checks=0;
 integer read_left=0,write_left=0,bdelay=0;reg [31:0] read_address=0,write_address=0;
 reg [31:0] rng=32'h23641871;reg held_ar=0,held_aw=0,held_w=0;
 reg [39:0] saved_ar,saved_aw;reg [144:0] saved_w;
 function [31:0] next_rng(input [31:0] a);reg [31:0] z;begin z=a^(a<<13);z=z^(z>>17);next_rng=z^(z<<5);end endfunction
 task fail(input string why);begin
  $display("FAIL PRODUCTION %s ticks=%0d computes=%0d dmas=%0d bytes=%0d core_fault=%h issue_fault=%h",why,ticks,compute_completed,dma_completed,output_bytes,fault_code,program_fault_code);
  $display("FAULT_DETAIL fetch=%b/%h issue=%b causes=%h core=%b header=%b dma=%b arb=%b route=%b store=%b cfg=%b/%b",dut.fetch_error,dut.fetch_fault,dut.issue_error,dut.core.causes,dut.core.core_error,dut.herror,dut.derror,dut.core.arb_error,dut.core.arena_router_error,store_error,dut.cfg_valid,dut.cfg_ready);$fatal(1);
 end endtask
 function [127:0] memory(input [31:0] address);
  integer offset;reg [127:0] value;begin
   value=0;
   if(address>=32'h04000000 && address<32'h04216600)begin
    offset=int'(address-32'h04000000);value=compute_image[offset/64][(offset%64)*8+:128];
   end else if(address>=32'h04300000 && address<32'h04370800)begin
    offset=int'(address-32'h04300000);value=dma_image[offset/16];
   end else if(address<2764800)begin
    for(integer b=0;b<16;b=b+1)value[b*8+:8]=source_image[address+b];
   end else $fatal(1,"out-of-window AXI read %h",address);
   memory=value;
  end
 endfunction
 always @(negedge clk)begin
  rng=next_rng(rng);
  arready=!reset && read_left==0 && !rvalid && rng[0];
  awready=!reset && write_left==0 && !bvalid && bdelay==0 && rng[1];
  wready=!reset && write_left!=0 && rng[2];
  if(!reset && !rvalid && read_left!=0 && rng[3])begin rdata=memory(read_address);rvalid=1;rlast=read_left==1;end
  if(bdelay!=0)begin bdelay=bdelay-1;if(bdelay==0)bvalid=1;end
 end
 always @(posedge clk)if(!reset)begin
  ticks=ticks+1;
  if(dut.cfg_valid&&dut.cfg_ready && compute_completed<2)$display("FIRST_CONFIG idx=%0d groups=%0d/%0d pixels=%0d rows=%0d planes=%h bases=%h out=%h weight=%0d",dut.debug_program_index,dut.cfg_in_groups,dut.cfg_groups,dut.cfg_pixels,dut.cfg_rows,dut.cfg_row_planes,dut.cfg_row_bases,dut.out_bases,dut.cfg_weight_offset);
  if(held_ar)begin if(!arvalid || {araddr,arlen}!==saved_ar)fail("held AR changed");held_checks=held_checks+1;end
  if(held_aw)begin if(!awvalid || {awaddr,awlen}!==saved_aw)fail("held AW changed");held_checks=held_checks+1;end
  if(held_w)begin if(!wvalid || {wdata,wstrb,wlast}!==saved_w)fail("held W changed");held_checks=held_checks+1;end
  held_ar=arvalid&&!arready;held_aw=awvalid&&!awready;held_w=wvalid&&!wready;
  saved_ar={araddr,arlen};saved_aw={awaddr,awlen};saved_w={wdata,wstrb,wlast};
  if(arvalid&&arready)begin
   if(araddr[3:0] || int'(araddr[11:0])+(int'(arlen)+1)*16>4096)fail("read burst boundary");
   read_address=araddr;read_left=int'(arlen)+1;reads_seen=reads_seen+1;
  end
  if(rvalid&&rready)begin rvalid<=0;read_address=read_address+16;read_left=read_left-1;end
  if(awvalid&&awready)begin
   if(awaddr[3:0] || int'(awaddr[11:0])+(int'(awlen)+1)*16>4096)fail("write burst boundary");
   write_address=awaddr;write_left=int'(awlen)+1;
  end
  if(wvalid&&wready)begin
   if(wlast!==(write_left==1))fail("WLAST");
   for(integer b=0;b<16;b=b+1)if(wstrb[b])begin
    integer off,y,x,index;off=int'(write_address)+b-'h800000;y=off/3840;x=off%3840;index=y*192+x;
    if(y<0 || y>=8 || x>=192)fail("output outside bounded 64x8 prefix");
    if(seen[index])fail("duplicate output byte");
    if(wdata[b*8+:8]!==golden[index])begin $display("RGB_MISMATCH y=%0d byte_x=%0d actual=%h gold=%h",y,x,wdata[b*8+:8],golden[index]);fail("RGB numerical mismatch");end
    seen[index]=1;output_bytes=output_bytes+1;
   end
   write_left=write_left-1;write_address=write_address+16;
   if(wlast)bdelay=2+int'(rng[7:5]);
  end
  if(bvalid&&bready)begin bvalid<=0;output_rows=output_rows+1;end
  if(!cancel && (error || frame_valid))fail("error or premature frame publication");
  if(ticks>500000)fail("bounded watchdog");
 end
 initial begin
  $readmemh("load_words.hex",coefficients);$readmemh("compute.hex",compute_image);$readmemh("dma.hex",dma_image);
  $readmemh("source.hex",source_image);$readmemh("golden.hex",golden);
  reset=1;repeat(5)@(negedge clk);reset=0;repeat(2)@(negedge clk);
  begin_version=1;begin_image_bytes=17792;begin_tag=16'h4321;job_image_tag=16'h4321;
  begin_valid=1;do @(posedge clk);while(!begin_ready);@(negedge clk);begin_valid=0;
  for(integer k=0;k<2560;k=k+1)begin
   load_index=12'(k);load_data=coefficients[k];load_last=k==2559;load_valid=1;
   do @(posedge clk);while(!load_ready);@(negedge clk);load_valid=0;
  end
  verify_tag=16'h4321;verify_ok=1;verify_valid=1;do @(posedge clk);while(!verify_ready);@(negedge clk);verify_valid=0;
  if(!coeff_ready || store_error)fail("coefficient load");
  compute_crc32=COMPUTE_CRC;dma_crc32=DMA_CRC;
  start_valid=1;do @(posedge clk);while(!start_ready);@(negedge clk);start_valid=0;
  while(output_rows<8)@(negedge clk);
  if(output_bytes!=1536 || !(&seen) || held_checks==0 || compute_completed<13)fail("coverage");
  // Cancel is an intentional short-test boundary, not a successful frame.
  cancel=1;while(!quiesced)@(negedge clk);
  if(frame_valid || read_left!=0 || write_left!=0 || rvalid || bvalid)fail("cancel did not drain");
  $display("PASS REAL_REGION_PRODUCTION_PREFIX pixels=512 bytes=%0d rows=%0d compute=%0d dma=%0d ticks=%0d held=%0d read_bursts=%0d",output_bytes,output_rows,compute_completed,dma_completed,ticks,held_checks,reads_seen);
  $finish;
 end
 initial begin #6000000;fail("global watchdog");end
endmodule
'''


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--program',type=Path,required=True)
    ap.add_argument('--issuer',type=Path);ap.add_argument('--fetch',type=Path);ap.add_argument('--rows',type=int,default=8);ap.add_argument('--full-frame',action='store_true');a=ap.parse_args();assert 1<=a.rows<=720
    if a.full_frame:a.rows=720
    root=ROOT/'runs'/RUN/a.name;root.mkdir(exist_ok=False);files=[];records=[];base=ROOT/'runs'/RUN
    manifest=json.loads((a.program/'program.json').read_text());cs,ds=manifest['streams'];bits=manifest.get('index_bits',16);width=manifest.get('stripe_width',64)
    if a.full_frame:width=1280
    def copy(f):
        dst=root/f.name;shutil.copy2(f,dst);files.append(dst);records.append(dict(source=str(f),source_sha256=file_hash(f),snapshot=str(dst)));return dst
    for f in (base/'rtl-variable-groups/frozen').iterdir():
        if f.suffix in ('.v','.sv') and not f.name.startswith('tb_'):copy(f)
    for name in ['next_header_prefetch.sv','resident_read_arbiter2.sv','arena_client_router2.sv','axi_read_merge2.sv','cnn_frame_datapath.sv','cnn_production_frame.sv']:copy(HW/name)
    copy((a.issuer or base/'rtl-region-issue-v1')/'frame_program_issue.sv');copy((a.fetch or base/'rtl-region-fetch-v1')/'frame_program_fetch.sv')
    copy(base/'rtl-region-dma-v2/frame_arena_row_dma.sv');copy(Path(__file__).with_name('region_luma_decode.sv'))
    f=root/'cnn_frame_datapath.sv';s=f.read_text();s=s.replace('input wire dstore,drgb,','input wire dstore,drgb,dregion,')
    s=s.replace('dma_bytes=drgb ?', 'dma_bytes=(drgb || dregion) ?').replace('.cmd_rgb(drgb),','.cmd_rgb(drgb),.cmd_region(dregion),');f.write_text(s)
    f=root/'cnn_production_frame.sv';s=f.read_text().replace('[13:0] compute_completed',f'[{bits-1}:0] compute_completed').replace('[14:0] debug_program_index',f'[{bits-1}:0] debug_program_index')
    s=s.replace("14'd8789",f"{bits}'d{cs['count']}").replace("14'd10440",f"{bits}'d{ds['count']}").replace('wire  drgb;','wire  drgb;wire dregion;').replace('.drgb(drgb),','.drgb(drgb),.dregion(dregion),');f.write_text(s)
    sys.path.insert(0,str(HW));from build_production_cosim import ports_of
    _,ports=ports_of(s);tb='module tb_region_production;\n'
    for d,w,n in ports:tb+=f'{"reg" if d=="input" else "wire"} [{w-1}:0] {n}'+('=0;\n' if d=='input' else ';\n')
    tb+='cnn_production_frame dut(.*);always #5 clk=~clk;\n'
    for name in ['load_words.hex','compute.hex','dma.hex']:shutil.copy2(a.program/name,root/name)
    m=json.loads((base/'rtl-graph-dma-v1/inputs.json').read_text());source=Path(m['image']);assert file_hash(source)==m['image_sha256']
    original=np.asarray(Image.open(source).convert('RGB'));package_path=Path(manifest['package']);assert file_hash(package_path)==manifest['package_sha256']
    package=json.loads(package_path.read_text());from .probe_region_class_int8 import execute
    # 128 original rows retain far more than the receptive-field halo needed
    # for the first eight outputs. Oracle is independent int64 CPU execution.
    height=min(720,max(128,(a.rows+67)//4*4))
    values=execute(package,(original[:height].astype(np.int64)-128).transpose(2,0,1)[None],False)['rgb'][0,:,:a.rows,:width]
    y,co,cg=[np.asarray(levels,np.int64)[values[s].argmax(0)] for levels,s in zip(package['levels'],[slice(0,16),slice(16,24),slice(24,32)])]
    gold=np.stack([y+co-cg,y+cg,y-co-cg],-1).clip(0,255).astype(np.uint8)
    reference=root/'golden-int64.png';Image.fromarray(gold).save(reference)
    assert original.shape==(720,1280,3) and gold.shape==(a.rows,width,3)
    for name,arr in [('source',original),('golden',gold)]:
        (root/(name+'.hex')).write_text(''.join(f'{int(b)^128:02x}\n' for b in arr.reshape(-1)),encoding='ascii')
    body=BODY.replace('COMPUTE_CRC',f"32'h{cs['crc32']:08x}").replace('DMA_CRC',f"32'h{ds['crc32']:08x}")
    replacements={'34199':str(cs['count']-1),'28799':str(ds['count']-1),'04216600':f"{cs['address']+cs['bytes']:08x}",
        '04300000':f"{ds['address']:08x}",'04370800':f"{ds['address']+ds['bytes']:08x}",'1535':str(width*a.rows*3-1),'1536':str(width*a.rows*3),
        '192':str(width*3),'pixels=512':f'pixels={width*a.rows}','64x8':f'{width}x{a.rows}',
        'y>=8':f'y>={a.rows}','output_rows<8':f'output_rows<{a.rows}','ticks>500000':'ticks>2000000','#6000000;':'#21000000;'}
    for old,new in replacements.items():body=body.replace(old,new)
    if a.full_frame:
        # Unpacked seen bits keep generated C++ bounded instead of one huge
        # constant-width reduction. Every actual RGB byte is still checked.
        total=720*1280*3
        body=body.replace(f'reg [{total-1}:0] seen=0;',f'reg seen[0:{total-1}];integer active_start=0;')
        body=body.replace('  reset=1;repeat(5)',f'  for(integer j=0;j<{total};j=j+1)seen[j]=0;\n  reset=1;repeat(5)')
        body=body.replace('if(!cancel && (error || frame_valid))','if(!cancel && error)')
        body=body.replace('ticks>2000000','ticks>15000000').replace('#21000000;','#151000000;')
        body=body.replace('  start_valid=1;do','  active_start=ticks;start_valid=1;do')
        begin=body.index('  while(output_rows<720)');end=body.index('  $finish;',begin)
        body=body[:begin]+f'''  while(!frame_valid)@(negedge clk);
  if(output_bytes!={total} || held_checks==0 || compute_completed!={cs['count']} || dma_completed!={ds['count']})fail("full coverage/counts");
  if(nodes_completed!=29'h{manifest['complete_node_mask']:x} || rgb_pixels!=921600)fail("full graph commit");
  for(integer j=0;j<{total};j=j+1)if(!seen[j])fail("missing RGB byte");
  if(read_left!=0 || write_left!=0 || rvalid || bvalid || busy)fail("published with pending transactions");
  $display("PASS REAL_REGION_PRODUCTION_FRAME pixels=921600 bytes=%0d rows=%0d compute=%0d dma=%0d ticks=%0d active_ticks=%0d held=%0d read_bursts=%0d",output_bytes,output_rows,compute_completed,dma_completed,ticks,ticks-active_start,held_checks,reads_seen);
'''+body[end:]
    tbpath=root/'tb_region_production.sv';tbpath.write_text(tb+body);files.append(tbpath)
    write_json(root/'inputs.json',dict(program=str(a.program),program_sha256=file_hash(a.program/'program.json'),source=str(source),source_sha256=file_hash(source),gold=str(reference),gold_sha256=file_hash(reference),inputs=records,
        compiled_hashes={str(f):file_hash(f) for f in files},production_modified=False,no_intermediate_injection=True,rows=a.rows,width=width,oracle_height=height,full_frame=a.full_frame))
    build(root,files,'tb_region_production');seconds=run([root/'obj/Vtb_region_production.exe'],root,root/'simulation.log',3300 if a.full_frame else 600)
    label='FRAME' if a.full_frame else 'PREFIX'
    log=(root/'simulation.log').read_text();assert 'PASS REAL_REGION_PRODUCTION_'+label in log
    write_json(root/'result.json',dict(status='REAL_FETCH_ISSUER_HEADER_CORE_DMA_'+label+'_PASS',seconds=seconds,log=log,full_frame=a.full_frame,actual_DDR=False,fit=False,timing=False,board=False))
    print(log,flush=True)


if __name__=='__main__':main()
