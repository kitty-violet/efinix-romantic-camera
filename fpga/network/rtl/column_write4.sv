// Shared-pool results return through actual arena B ports. Fixed destinations
// plus a per-bank held-owner lock preserve valid/data during partial backpressure.
// Callers publish a block only after all expected write ACKs, never at RQ output.
module column_write4(
 input wire clk,reset,cancel,
 input wire [8:0] pixels,input wire [2:0] rows,groups,
 input wire [7:0] expected_mask,row_planes,input wire [39:0] row_bases,
 input wire [3:0] in_valid,output reg [3:0] in_ready,
 input wire [255:0] in_data,input wire [63:0] in_tag,
 input wire [31:0] in_mask,in_bias_overflow,in_range_error,
 input wire [23:0] read_locked,
 output wire [23:0] mem_req_valid,input wire [23:0] mem_req_ready,
 output wire [239:0] mem_req_address,output wire [1535:0] mem_req_data,
 output wire [383:0] mem_req_tag,
 input wire [23:0] mem_rsp_valid,output wire [23:0] mem_rsp_ready,
 input wire [383:0] mem_rsp_tag,input wire [23:0] mem_rsp_was_write,
 output wire busy,output reg error,output reg [31:0] writes,acks
);
 wire active=!reset && !cancel && !error;
 wire [3:0] target_bank[0:3];wire [9:0] target_address[0:3];wire [3:0] bad;
 reg [11:0] held,pending;reg [1:0] held_owner[0:11];reg [15:0] expected[0:11];
 wire [1:0] chosen[0:11];wire [11:0] valid,fire;
 assign busy=(|pending)||(|held);
 for(genvar o=0;o<4;o=o+1)begin:RESULT
  wire [15:0] tag=in_tag[o*16+:16];wire [1:0] y=tag[12:11],g=tag[1:0];wire [8:0] x=tag[10:2];
  wire [11:0] off=groups==4 ? {3'd0,x[8:2],2'd0} : groups==2 ? {4'd0,x[8:2],1'b0} : {5'd0,x[8:2]};
  wire [12:0] address={3'd0,row_bases[y*10+:10]}+{1'b0,off}+{11'd0,g};
  assign target_bank[o]={row_planes[y*2+:2],x[1:0]};assign target_address[o]=address[9:0];
  assign bad[o]=in_valid[o] && (tag[15:13]!=0 || {1'b0,y}>=rows || x>=pixels || {1'b0,g}>=groups ||
   row_planes[y*2+:2]>=3 || address>=1024 || in_mask[o*8+:8]!=expected_mask || (|in_bias_overflow[o*8+:8]) || (|in_range_error[o*8+:8]));
 end
 for(genvar b=0;b<12;b=b+1)begin:BANK
  reg found;reg [1:0] choice;
  always @*begin
   found=0;choice=held_owner[b];
   if(held[b])found=in_valid[choice];
   else for(integer o=0;o<4;o=o+1)if(!found && in_valid[o] && target_bank[o]==4'(b))begin found=1;choice=2'(o);end
  end
  assign chosen[b]=choice;assign valid[b]=active && found && !read_locked[b*2+1] && !(|bad);
  assign mem_req_valid[b*2+:2]={valid[b],1'b0};
  assign mem_req_address[b*20+:20]={target_address[choice],10'd0};
  assign mem_req_data[b*128+:128]={in_data[choice*64+:64],64'd0};
  assign mem_req_tag[b*32+:32]={in_tag[choice*16+:16],16'd0};
  assign mem_rsp_ready[b*2+:2]={2{active}};assign fire[b]=valid[b] && mem_req_ready[b*2+1];
  always @(posedge clk)begin
   if(reset || cancel)begin held[b]<=0;held_owner[b]<=0;pending[b]<=0;expected[b]<=0;end
   else if(active)begin
    if(valid[b] && !mem_req_ready[b*2+1])begin held[b]<=1;held_owner[b]<=choice;end
    if(mem_rsp_valid[b*2+1])pending[b]<=0;
    if(fire[b])begin held[b]<=0;pending[b]<=1;expected[b]<=in_tag[choice*16+:16];end
   end
  end
 end
 integer write_now,ack_now;
 always @*begin
  in_ready=0;write_now=0;ack_now=0;
  for(integer b=0;b<12;b=b+1)begin
   if(fire[b])begin in_ready[chosen[b]]=1;write_now=write_now+1;end
   if(mem_rsp_valid[b*2+1] && mem_rsp_ready[b*2+1])ack_now=ack_now+1;
  end
 end
 always @(posedge clk)begin
  if(reset || cancel)begin error<=0;writes<=0;acks<=0;end
  else if(active)begin
   writes<=writes+32'(write_now);acks<=acks+32'(ack_now);
   if(|bad)error<=1;
   for(integer b=0;b<12;b=b+1)begin
    if(mem_rsp_valid[b*2] || (mem_rsp_valid[b*2+1] && (!pending[b] || !mem_rsp_was_write[b*2+1] || mem_rsp_tag[(b*2+1)*16+:16]!=expected[b])))error<=1;
   end
  end
 end
endmodule
