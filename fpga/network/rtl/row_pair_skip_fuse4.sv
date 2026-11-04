// Fixed C32 residual merge. Each of four rows owns one distinct row bank.
// The real RAM retains the skip return through RQ backpressure; no private
// four-word skip copy and no twelve-port tagged read owner crossbar.
module row_pair_skip_fuse4(
 input wire clk,reset,cancel,enable,main_scaled,
 input wire [8:0] pixels,input wire [2:0] rows,groups,
 input wire [11:0] skip_banks,input wire [39:0] skip_bases,input wire [3:0] skip_phase,input wire [10:0] skip_delta,
 input wire [3:0] mac_valid,output wire [3:0] mac_ready,input wire [255:0] mac_data,input wire [63:0] mac_tag,
 input wire [31:0] mac_mask,mac_bias_overflow,mac_range_error,
 output wire [3:0] bypass_valid,input wire [3:0] bypass_ready,
 output wire [287:0] bypass_value,output wire [63:0] bypass_tag,output wire [31:0] bypass_mask,
 input wire [1:0] aligned_valid,output wire [1:0] aligned_ready,input wire [127:0] aligned_data,input wire [31:0] aligned_tag,
 input wire [15:0] aligned_mask,aligned_bias_overflow,aligned_range_error,
 output wire [5:0] mem_req_valid,input wire [5:0] mem_req_ready,output wire [59:0] mem_req_address,
 input wire [5:0] mem_rsp_valid,output wire [5:0] mem_rsp_ready,input wire [767:0] mem_rsp_data,input wire [5:0] mem_rsp_was_write,
 output wire busy,output reg error,output reg [31:0] paired_words
);
 reg [3:0] pending,held,half_q;reg [63:0] main_q[0:3];reg [15:0] tag_q[0:3];
 reg [1:0] prefer,align_held,align_owner;wire [1:0] chosen;wire [3:0] align_take,have;
 wire [1:0] qvalid,qempty,qerror,pair_bad;wire active=enable && !reset && !cancel && !error;
 wire [3:0] bad,offer,accept;wire [2:0] bank[0:3];wire [9:0] address[0:3];wire [3:0] target_half;
 reg config_bad;
 always @*begin
  config_bad=groups!=4 || (skip_delta!=0 && skip_delta!=2) || skip_phase!=0 || rows==0 || rows>4 || pixels==0 || pixels>320;
  for(integer a=0;a<4;a=a+1)if(a<rows)begin
   if(skip_banks[a*3+:3]>=6)config_bad=1;
   for(integer b=0;b<a;b=b+1)if(skip_banks[a*3+:3]==skip_banks[b*3+:3])config_bad=1;
  end
 end
 assign busy=(|pending)||(|held)||!( &qempty);
 for(genvar b=0;b<4;b=b+1)begin:ROW
  wire [15:0] tag=mac_tag[b*16+:16];wire [1:0] y=tag[12:11],g=tag[1:0];wire [8:0] x=tag[10:2];
  wire [10:0] sx={2'd0,x}+{9'd0,skip_delta[1],1'b0};
  wire [12:0] addr={3'd0,skip_bases[b*10+:10]}+{1'b0,sx[10:1],2'd0}+{11'd0,g};
  assign bank[b]=skip_banks[b*3+:3];assign address[b]=addr[9:0];assign target_half[b]=sx[0];
  assign bad[b]=mac_valid[b] && (config_bad || tag[15:13]!=0 || y!=2'(b) || {1'b0,y}>=rows || x>=pixels ||
   addr>=1024 || mac_mask[b*8+:8]!=8'hff || (|mac_bias_overflow[b*8+:8]) || (|mac_range_error[b*8+:8]));
  assign have[b]=active && pending[b] && mem_rsp_valid[bank[b]];
  assign offer[b]=active && !(|bad) && mac_valid[b] && (held[b] || !pending[b] || align_take[b]);
  assign accept[b]=offer[b] && mem_req_ready[bank[b]];
  assign mac_ready[b]=accept[b];
 end
 for(genvar p=0;p<6;p=p+1)begin:PORT
  reg valid,ready;reg [9:0] a;
  always @*begin
   valid=0;ready=0;a=0;
   for(integer b=0;b<4;b=b+1)if(bank[b]==3'(p) && b<rows)begin
    valid=offer[b];a=address[b];ready=align_take[b];
   end
  end
  assign mem_req_valid[p]=valid;assign mem_req_address[p*10+:10]=a;assign mem_rsp_ready[p]=ready;
 end
 for(genvar b=0;b<2;b=b+1)begin:ALIGN
  wire select_hi=align_held[b] ? align_owner[b] : (have[b+2] && (!have[b] || prefer[b]));
  wire [1:0] owner={select_hi,1'(b)};wire valid=have[owner];
  wire [2:0] selected_bank=skip_banks[owner*3+:3];
  wire [63:0] selected_skip=mem_rsp_data[selected_bank*128+half_q[owner]*64+:64];
  assign chosen[b]=select_hi;
  assign bypass_valid[b]=active && valid;
  wire take=bypass_valid[b] && bypass_ready[b];
  assign align_take[b]=take && !select_hi;assign align_take[b+2]=take && select_hi;
  assign bypass_tag[b*16+:16]=tag_q[owner];assign bypass_mask[b*8+:8]=8'hff;
  wire [79:0] qdata;
  wire pop=aligned_valid[b] && aligned_ready[b];
  shallow_result_fifo #(.WIDTH(80)) unscaled(.clk(clk),.reset(reset || cancel),.push(take),
   .in_data({tag_q[owner],(main_scaled ? selected_skip:main_q[owner])}),
   .out_valid(qvalid[b]),.out_ready(pop),.out_data(qdata),.empty(qempty[b]),.error(qerror[b]));
  assign pair_bad[b]=aligned_valid[b] && qvalid[b] && (aligned_tag[b*16+:16]!=qdata[79:64] || aligned_mask[b*8+:8]!=8'hff ||
   (|aligned_bias_overflow[b*8+:8]) || (|aligned_range_error[b*8+:8]));
  assign bypass_valid[b+2]=active && aligned_valid[b] && qvalid[b] && !pair_bad[b];
  assign aligned_ready[b]=active && qvalid[b] && bypass_ready[b+2] && !pair_bad[b];
  assign bypass_tag[(b+2)*16+:16]=qdata[79:64];assign bypass_mask[(b+2)*8+:8]=8'hff;
  for(genvar l=0;l<8;l=l+1)begin:LANE
   wire signed [7:0] v=main_scaled ? main_q[owner][l*8+:8]:selected_skip[l*8+:8];
   wire signed [7:0] a=aligned_data[(b*8+l)*8+:8],c=qdata[l*8+:8];
   assign bypass_value[(b*8+l)*9+:9]={v[7],v};
   assign bypass_value[((b+2)*8+l)*9+:9]=$signed({a[7],a})+$signed({c[7],c});
  end
  always @(posedge clk)begin
   if(reset || cancel)begin prefer[b]<=0;align_held[b]<=0;align_owner[b]<=0;end
   else if(active)begin
    if(bypass_valid[b] && !bypass_ready[b])begin align_held[b]<=1;align_owner[b]<=select_hi;end
    if(take)begin align_held[b]<=0;prefer[b]<=!select_hi;end
   end
  end
 end

 always @(posedge clk)begin
  if(reset || cancel)begin
   pending<=0;held<=0;half_q<=0;error<=0;paired_words<=0;
   for(integer b=0;b<4;b=b+1)begin main_q[b]<=0;tag_q[b]<=0;end
  end else if(active)begin
   if((|bad)||(|pair_bad)||(|qerror))error<=1;
   paired_words<=paired_words+32'(bypass_valid[2] && bypass_ready[2])+32'(bypass_valid[3] && bypass_ready[3]);
   for(integer b=0;b<4;b=b+1)begin
    if(align_take[b])pending[b]<=0;
    if(offer[b] && !accept[b])held[b]<=1;
    if(held[b] && !mac_valid[b])error<=1;
    if(accept[b])begin pending[b]<=1;held[b]<=0;half_q[b]<=target_half[b];main_q[b]<=mac_data[b*64+:64];tag_q[b]<=mac_tag[b*16+:16];end
   end
   for(integer p=0;p<6;p=p+1)if(mem_rsp_valid[p])begin
    reg owner;owner=0;
    for(integer b=0;b<4;b=b+1)if(bank[b]==3'(p) && pending[b])owner=1;
    if(!owner || mem_rsp_was_write[p])error<=1;
   end
  end
 end
endmodule
