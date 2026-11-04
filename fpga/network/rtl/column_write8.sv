// Four packed RQ sources connect directly to the twelve arena B ports.
// Keep eight independent pixel acceptance bits, but select packed data,
// address ONCE from four sources, before unpacking an RGB pair.
// This replaces result_pair_expand4 -> eight-source column_write8.
// Port ownership, read locks, ACK checks and cancellation are unchanged.
module column_write8(
 input wire clk,reset,cancel,pair_mode,
 input wire [8:0] pixels,input wire [2:0] rows,groups,
 input wire [7:0] expected_mask,row_planes,input wire [39:0] row_bases,
 input wire [3:0] in_valid,output wire [3:0] in_ready,
 input wire [255:0] in_data,input wire [63:0] in_tag,
 input wire [31:0] in_mask,in_bias_overflow,in_range_error,
 input wire [23:0] read_locked,
 output wire [23:0] mem_req_valid,input wire [23:0] mem_req_ready,
 output wire [239:0] mem_req_address,output wire [1535:0] mem_req_data,
 output wire [383:0] mem_req_tag,
 input wire [23:0] mem_rsp_valid,output wire [23:0] mem_rsp_ready,
 input wire [383:0] mem_rsp_tag,input wire [23:0] mem_rsp_was_write,
 output wire busy,error,output reg [31:0] writes,acks
);
 reg expand_fault,write_fault;
 reg [7:0] taken;
 wire expand_active=!reset && !cancel && !expand_fault;
 wire active=!reset && !cancel && !write_fault;
 wire [3:0] pair_bad;
 wire [7:0] valid_pixel,bad;
 reg [7:0] ready_pixel;
 wire [3:0] target_bank[0:7];
 wire [9:0] source_address[0:3];
 wire [39:0] compact_address;
 reg [11:0] held,pending;
 reg [2:0] held_owner[0:11];
 wire [2:0] chosen[0:11];wire [11:0] valid,fire;
 assign busy=(|pending)||(|held)||(|taken);
 assign error=expand_fault || write_fault;
 for(genvar s=0;s<4;s=s+1)begin:SOURCE
  wire [15:0] tag=in_tag[s*16+:16];
  wire [1:0] y=tag[12:11],g=tag[1:0];
  wire [9:0] pair_x={tag[10:2],1'b0};
  wire [8:0] x=pair_mode ? pair_x[8:0]:tag[10:2];
  wire [1:0] group_q=pair_mode ? 2'd0:g;
  wire second=pair_mode && pair_x+10'd1<{1'b0,pixels};
  wire [1:0] plane=row_planes[y*2+:2];
  wire [11:0] off=groups==4 ? {3'd0,x[8:2],2'd0}:groups==2 ? {4'd0,x[8:2],1'b0}:{5'd0,x[8:2]};
  wire [12:0] address={3'd0,row_bases[y*10+:10]}+{1'b0,off}+{11'd0,group_q};
  assign source_address[s]=address[9:0];
  assign compact_address[s*10+:10]=source_address[s];
  assign pair_bad[s]=in_valid[s] && pair_mode && (tag[15:13]!=0 || g!=0 || pair_x>={1'b0,pixels} ||
   in_mask[s*8+:8]!=8'h3f || (|in_bias_overflow[s*8+:8]) || (|in_range_error[s*8+:8]));
  wire accepted_source=expand_active && !pair_bad[s];
  assign valid_pixel[s*2]=accepted_source && in_valid[s] && !taken[s*2];
  assign valid_pixel[s*2+1]=accepted_source && in_valid[s] && second && !taken[s*2+1];
  assign in_ready[s]=accepted_source && (taken[s*2] || ready_pixel[s*2]) &&
   (!second || taken[s*2+1] || ready_pixel[s*2+1]);
  assign target_bank[s*2]={plane,x[1:0]};
  // A pair's first pixel is even, so its second pixel never crosses a
  // four-column address boundary. Both use the same compact address.
  assign target_bank[s*2+1]={plane,x[1],1'b1};
  wire common_bad={1'b0,y}>=rows || {1'b0,group_q}>=groups || plane>=3 || address>=1024 ||
   (pair_mode ? expected_mask!=8'h07 :
    (tag[15:13]!=0 || in_mask[s*8+:8]!=expected_mask || (|in_bias_overflow[s*8+:8]) || (|in_range_error[s*8+:8])));
  assign bad[s*2]=valid_pixel[s*2] && (common_bad || x>=pixels);
  assign bad[s*2+1]=valid_pixel[s*2+1] && (common_bad || groups!=1 || {1'b0,x}+10'd1>={1'b0,pixels});
  always @(posedge clk)begin
   if(reset || cancel)taken[s*2+:2]<=0;
   else if(in_valid[s] && in_ready[s])taken[s*2+:2]<=0;
   else taken[s*2+:2]<=taken[s*2+:2]|(valid_pixel[s*2+:2]&ready_pixel[s*2+:2]);
  end
 end
 for(genvar b=0;b<12;b=b+1)begin:BANK
  reg found;reg [2:0] choice;
  always @*begin
   found=0;choice=held_owner[b];
   if(held[b])found=valid_pixel[choice];
   else for(integer o=0;o<8;o=o+1)if(!found && valid_pixel[o] && target_bank[o]==4'(b))begin found=1;choice=3'(o);end
  end
  assign chosen[b]=choice;
  assign valid[b]=active && found && !read_locked[b*2+1] && !(|bad);
  assign mem_req_valid[b*2+:2]={valid[b],1'b0};
  wire [63:0] packed_word=in_data[choice[2:1]*64+:64];
  wire [63:0] write_word=pair_mode ? {40'd0,packed_word[(b%2)*24+:24]}:packed_word;
  assign mem_req_address[b*20+:20]={compact_address[choice[2:1]*10+:10],10'd0};
  assign mem_req_data[b*128+:128]={write_word,64'd0};
  assign mem_req_tag[b*32+:32]=32'd0;
  assign mem_rsp_ready[b*2+:2]={2{active}};
  assign fire[b]=valid[b] && mem_req_ready[b*2+1];
  always @(posedge clk)begin
   if(reset || cancel)begin held[b]<=0;held_owner[b]<=0;pending[b]<=0;end
   else if(active)begin
    if(valid[b] && !mem_req_ready[b*2+1])begin held[b]<=1;held_owner[b]<=choice;end
    if(mem_rsp_valid[b*2+1])pending[b]<=0;
    if(fire[b])begin held[b]<=0;pending[b]<=1;end
   end
  end
 end
 integer write_now,ack_now;
 always @*begin
  ready_pixel=0;write_now=0;ack_now=0;
  for(integer b=0;b<12;b=b+1)begin
   if(fire[b])begin ready_pixel[chosen[b]]=1;write_now=write_now+1;end
   if(mem_rsp_valid[b*2+1] && mem_rsp_ready[b*2+1])ack_now=ack_now+1;
  end
 end
 always @(posedge clk)begin
  if(reset || cancel)begin expand_fault<=0;write_fault<=0;writes<=0;acks<=0;end
  else begin
   if(|pair_bad)expand_fault<=1;
   if(active)begin
    writes<=writes+32'(write_now);acks<=acks+32'(ack_now);
    if(|bad)write_fault<=1;
    for(integer b=0;b<12;b=b+1)begin
     if(mem_rsp_valid[b*2] || (mem_rsp_valid[b*2+1] && (!pending[b] || !mem_rsp_was_write[b*2+1])))write_fault<=1;
    end
   end
  end
 end
endmodule
