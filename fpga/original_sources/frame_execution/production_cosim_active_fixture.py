"""Bounded real descriptor/CNN/AXI prefix stimulus for engine equivalence.

This is an external AXI behavioral memory, not the board's vendor DDR PHY.
The actual frozen board coefficient and descriptor packages are hash checked.
"""
import json
from pathlib import Path
from run_units import ROOT, sha


def active_fixture(tb,root):
    payload=ROOT/'tmp/frame-execution/board-payload2'
    meta=json.loads((payload/'payload.json').read_text())
    items={r['file']:r for r in meta['payloads']}
    paths={}
    for name in ['compute.bin','dma.bin','coefficients-padded.bin']:
        f=payload/name;assert sha(f)==items[name]['sha256']
        data=f.read_bytes();unit=8 if name.startswith('coefficients') else 16
        out=root/(name+'.hex');out.write_text(''.join(data[i:i+unit][::-1].hex()+'\n' for i in range(0,len(data),unit)))
        paths[name]=out.as_posix()
    dma_bytes=(payload/'dma.bin').read_bytes()
    first_store=next((i,int.from_bytes(dma_bytes[i*16:(i+1)*16],'little')&0x3fff)
                     for i in range(len(dma_bytes)//16) if (int.from_bytes(dma_bytes[i*16:(i+1)*16],'little')>>14)&1)
    # Reuse the verified load/readback task, now with the actual board image.
    old=ROOT/'tmp/frame-execution/pruned-unit1/frozen/vectors/load_words.hex'
    tb=tb.replace(old.as_posix(),paths['coefficients-padded.bin'])
    start=tb.index('initial begin\n $readmemh(')
    tb=tb[:start]+r'''
reg [127:0] compute_rom[0:35155],dma_rom[0:10439];
reg [127:0] memory[bit[31:0]];
reg [31:0] random_state=32'h189a7531,rd_address=0,wr_address=0;
integer rd_left=0,wr_left=0,ar_count=0,aw_count=0,r_count=0,w_count=0,b_count=0,ar_hold=0,r_hold=0;
reg rd_open=0,wr_open=0;reg [127:0] next_word;
function automatic [31:0] mix(input [31:0] x);reg[31:0] y;begin y=x^(x<<13);y=y^(y>>17);mix=y^(y<<5);end endfunction
function automatic [127:0] memread(input [31:0] address);
 reg[127:0] value;begin
  if(address>=32'h04000000 && address<32'h04089540)memread=compute_rom[(address-32'h04000000)>>4];
  else if(address>=32'h04100000 && address<32'h04128c80)memread=dma_rom[(address-32'h04100000)>>4];
  else if(memory.exists(address))memread=memory[address];
  else begin
   for(integer i=0;i<16;i++)value[i*8+:8]=8'(((address+i)*13)^((address+i)>>8));
   memread=value;
  end
 end
endfunction
// Endpoint state and outputs change by NBA, preserving the sampled handshake.
always @(posedge clk)begin
 if(reset)begin
  arready<=0;awready<=0;wready<=0;rvalid<=0;bvalid<=0;rlast<=0;rresp<=0;bresp<=0;
  rd_open<=0;wr_open<=0;rd_left<=0;wr_left<=0;
 end else begin
  random_state<=mix(random_state);
  arready<=!rd_open && !rvalid && random_state[0];
  awready<=!wr_open && !bvalid && random_state[1];
  wready<=wr_open && random_state[2];
  if(arvalid && !arready)ar_hold<=ar_hold+1;
  if(rvalid && !rready)r_hold<=r_hold+1;
  if(arvalid && arready)begin
   if(rd_open)$fatal(1,"behavior endpoint accepted overlapping read");
   rd_open<=1;rd_address<=araddr;rd_left<=int'(arlen)+1;ar_count<=ar_count+1;arready<=0;
  end
  if(rd_open && !rvalid && random_state[3])begin
   rdata<=memread(rd_address);rlast<=rd_left==1;rresp<=0;rvalid<=1;
  end
  if(rvalid && rready)begin
   r_count<=r_count+1;rvalid<=0;rd_address<=rd_address+16;rd_left<=rd_left-1;
   if(rd_left==1)rd_open<=0;
  end
  if(awvalid && awready)begin
   if(wr_open)$fatal(1,"behavior endpoint accepted overlapping write");
   wr_open<=1;wr_address<=awaddr;wr_left<=int'(awlen)+1;aw_count<=aw_count+1;awready<=0;
  end
  if(wvalid && wready)begin
   if(!wr_open || wlast!==(wr_left==1))$fatal(1,"write burst protocol mismatch");
   next_word=memread(wr_address);
   for(integer i=0;i<16;i++)if(wstrb[i])next_word[i*8+:8]=wdata[i*8+:8];
   memory[wr_address]=next_word;wr_address<=wr_address+16;wr_left<=wr_left-1;w_count<=w_count+1;
   if(wr_left==1)begin wr_open<=0;wready<=0;bvalid<=1;bresp<=0;end
  end
  if(bvalid && bready)begin bvalid<=0;b_count<=b_count+1;end
 end
end
initial begin
 $readmemh("COEFF",image);$readmemh("COMPUTE",compute_rom);$readmemh("DMA",dma_rom);
 reset_all();compute_crc32=32'h5c285c88;dma_crc32=32'hb3bc2b60;load_image();start_job();
 while(compute_completed<REQUIRED_COMPUTE || dma_completed<REQUIRED_DMA)begin
  @(negedge clk);if(error)$fatal(1,"active production prefix error fault=%h program=%h",fault_code,program_fault_code);
 end
 if(aw_count==0 || w_count==0 || b_count==0 || r_count==0 || ar_hold==0 || frame_valid)$fatal(1,"active prefix coverage incomplete compute=%0d dma=%0d AW=%0d W=%0d B=%0d R=%0d ARhold=%0d frame=%b",compute_completed,dma_completed,aw_count,w_count,b_count,r_count,ar_hold,frame_valid);
 $display("PASS PRODUCTION_DPI_ACTIVE_PREFIX compute=%0d dma=%0d AR=%0d R=%0d AW=%0d W=%0d B=%0d ARhold=%0d Rhold=%0d comparisons=%0d sim_ns=%0.3f",compute_completed,dma_completed,ar_count,r_count,aw_count,w_count,b_count,ar_hold,r_hold,comparisons,$realtime);
 $finish;
end
initial begin #1000000;$fatal(1,"active prefix watchdog");end
endmodule
'''.replace('REQUIRED_COMPUTE',str(first_store[1])).replace('REQUIRED_DMA',str(first_store[0]+1)).replace('COEFF',paths['coefficients-padded.bin']).replace('COMPUTE',paths['compute.bin']).replace('DMA',paths['dma.bin'])
    # CRCs are taken from the same packed payload, never guessed literals.
    tb=tb.replace("32'h5c285c88",f"32'h{items['compute.bin']['crc32']:08x}").replace("32'hb3bc2b60",f"32'h{items['dma.bin']['crc32']:08x}")
    for name,base,label in [('compute.bin',0x04000000,'04089540'),('dma.bin',0x04100000,'04128c80')]:
        tb=tb.replace("32'h"+label,f"32'h{base+items[name]['bytes']:08x}")
    (root/'active_payload.json').write_text(json.dumps(dict(payload_manifest_sha256=sha(payload/'payload.json'),items=items,
        first_store_dma_index=first_store[0],required_compute=first_store[1],required_dma=first_store[0]+1,
        scope='production prefix through first real store, external behavioral AXI memory, deterministic input; not PHY or full frame'),indent=2))
    return tb
