"""Bounded AXI sink for the real graph + region DMA integration bench."""

DECL=r'''
 wire [23:0] rat_unused;wire [383:0] rtag;
 wire [31:0] daraddr,dawaddr,dcycles,dwords,dbeats;wire [7:0] darlen,dawlen;
 wire darvalid,dawvalid,drready,dwlast,dwvalid,dbready;
 wire [127:0] dwdata;wire [15:0] dwstrb;reg dawready=0,dwready=0,dbvalid=0;
 reg [31:0] row_ddr_address;integer row_golden,row_size,ddr_pointer=0,ddr_left=0,bdelay=0,output_bytes=0,total_output_bytes=0;
 frame_arena_row_dma #(.ABORT_SHARED_ARENA(1)) output_dma(.clk(clk),.reset(reset),.cancel(cancel),
  .cmd_valid(rcv),.cmd_ready(rcr),.cmd_store(1'b1),.cmd_rgb(1'b0),.cmd_region(1'b1),
  .cmd_address(row_ddr_address),.cmd_columns(rpixels),.cmd_groups(3'd4),.cmd_plane(rsource),.cmd_base(rbase),
  .done_valid(rdv),.done_ready(rdr),.error(rerror),.busy(rbusy),.av(rv),.ar(ar),.aw(rw),.aa(ra),.ad(rd),.at(rtag),
  .sv(sv&{24{row_mode}}),.sr(rr),.sd(sd),.st(st),.sw(sw),
  .araddr(daraddr),.arlen(darlen),.arvalid(darvalid),.arready(1'b0),.rdata(128'd0),.rresp(2'd0),.rlast(1'b0),.rvalid(1'b0),.rready(drready),
  .awaddr(dawaddr),.awlen(dawlen),.awvalid(dawvalid),.awready(dawready),.wdata(dwdata),.wstrb(dwstrb),.wlast(dwlast),.wvalid(dwvalid),.wready(dwready),
  .bresp(2'd0),.bvalid(dbvalid),.bready(dbready),.cycles(dcycles),.arena_words(dwords),.axi_beats(dbeats));
 reg [31:0] d_rng=32'h872189;
 reg d_held_aw=0,d_held_w=0;reg [39:0] d_save_aw;reg [144:0] d_save_w;
 always @(negedge clk)begin
  d_rng=next_rng(d_rng);dawready=!reset && ddr_left==0 && !dbvalid && bdelay==0 && d_rng[0];
  dwready=!reset && ddr_left!=0 && d_rng[1];
  if(bdelay!=0)begin bdelay=bdelay-1;if(bdelay==0)dbvalid=1;end
 end
 always @(posedge clk)if(!reset)begin
  if(d_held_aw && (!dawvalid || {dawaddr,dawlen}!==d_save_aw))fail("DMA held AW");
  if(d_held_w && (!dwvalid || {dwdata,dwstrb,dwlast}!==d_save_w))fail("DMA held W");
  d_held_aw=dawvalid&&!dawready;d_held_w=dwvalid&&!dwready;d_save_aw={dawaddr,dawlen};d_save_w={dwdata,dwstrb,dwlast};
  if(darvalid)fail("output DMA unexpectedly reading DDR");
  if(dawvalid && dawready)begin
   if(dawaddr[3:0] || int'(dawaddr[11:0])+(int'(dawlen)+1)*16>4096)fail("DMA burst boundary");
   ddr_pointer=int'(dawaddr);ddr_left=int'(dawlen)+1;
  end
  if(dwvalid && dwready)begin
   if(dwlast!==(ddr_left==1))fail("DMA WLAST");
   for(integer b=0;b<16;b=b+1)begin
    integer idx;idx=ddr_pointer+b-int'(row_ddr_address);
    if(dwstrb[b] !== (idx>=0 && idx<row_size))fail("DMA byte strobe");
    if(dwstrb[b])begin
     if(dwdata[b*8+:8]!==golden[row_golden+idx/3][(idx%3)*8+:8])begin
      $display("DMA_PIXEL idx=%0d actual=%h expected=%h",idx,dwdata[b*8+:8],golden[row_golden+idx/3][(idx%3)*8+:8]);fail("DMA RGB output");end
     output_bytes=output_bytes+1;total_output_bytes=total_output_bytes+1;
    end
   end
   ddr_pointer=ddr_pointer+16;ddr_left=ddr_left-1;if(ddr_left==0)bdelay=17;
  end
  if(dbvalid && dbready)dbvalid<=0;
 end
'''

TASK=r'''
 task dma_row(input integer plane,base,n,golden_offset,ddr_address);
  begin run_mode=0;row_mode=1;rsource=2'(plane);rbase=10'(base);rpixels=9'(n);
   row_ddr_address=32'(ddr_address);row_golden=golden_offset;row_size=n*3;output_bytes=0;
   @(negedge clk);rcv=1;do @(posedge clk);while(!rcr);@(negedge clk);rcv=0;
   waits=0;while(!rdv)begin @(negedge clk);waits=waits+1;if(waits>10000 || rerror)fail("graph output DMA");end
   if(dwords!=32'(n*4) || output_bytes!=n*3 || ddr_left || dbvalid || bdelay)fail("output DMA completion/accounting");
   frame_cycles=frame_cycles+int'(dcycles);rdr=1;@(negedge clk);rdr=0;row_mode=0;
  end
 endtask
'''


def adapt(tasks):
    a=tasks.index(' region_row_decode reduction(');b=tasks.index(' reg [63:0] words',a)
    tasks=tasks[:a]+DECL+tasks[b:]
    a=tasks.index(' task decode_row');tasks=tasks[:a]+TASK
    return tasks
