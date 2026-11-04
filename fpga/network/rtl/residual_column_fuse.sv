// Two residual rows: reader words 0/1 are scaled/unscaled row0, 2/3 row1.
// RQ banks0/1 align; banks2/3 requantize the exact signed9 sum. The aligned
// intermediate never visits the tensor arena. Two actual16-word RAM FIFOs hold
// unscaled operands and share the alignment RQ credit count (push/pop together).
module residual_column_fuse(
 input wire clk,reset,cancel,
 input wire cmd_valid,output wire cmd_ready,input wire [8:0] cmd_pixels,
 input wire [1:0] cmd_rows,input wire [2:0] cmd_groups,
 input wire in_valid,output wire in_ready,input wire [383:0] in_data,input wire [12:0] in_index,
 output reg done_valid,input wire done_ready,output wire busy,output reg error,
 output wire [3:0] bypass_valid,input wire [3:0] bypass_ready,
 output wire [287:0] bypass_value,output wire [63:0] bypass_tag,output wire [31:0] bypass_mask,
 input wire [1:0] aligned_valid,output wire [1:0] aligned_ready,
 input wire [127:0] aligned_data,input wire [31:0] aligned_tag,
 input wire [15:0] aligned_mask,aligned_bias_overflow,aligned_range_error
);
 reg running;reg [8:0] pixels;reg [1:0] rows,lg;reg [2:0] groups;
 reg [12:0] next_index,total;reg [1:0] accepted;
 wire enable=!reset && !cancel && !error;
 wire [1:0] ready_lane,need;
 wire [10:0] x=11'((next_index>>({1'b0,lg}+3'd2))<<2)+{9'd0,next_index[1:0]};
 wire [1:0] group_index=next_index[3:2]&(groups[1:0]-2'd1);
 wire [1:0] qvalid,qempty,qerror,pair_bad;
 wire pop=in_valid && in_ready;
 assign busy=running || done_valid || (|accepted) || !( &qempty);
 assign cmd_ready=enable && !busy;
 assign in_ready=enable && running && (&ready_lane);
 for(genvar b=0;b<2;b=b+1)begin:PAIR
  assign need[b]=b<rows && x<{2'd0,pixels};
  assign ready_lane[b]=!need[b] || accepted[b] || bypass_ready[b];
  assign bypass_valid[b]=enable && running && in_valid && need[b] && !accepted[b];
  wire take=bypass_valid[b] && bypass_ready[b];
  wire [15:0] tag={3'd0,2'(b),x[8:0],group_index};
  assign bypass_tag[b*16+:16]=tag;assign bypass_mask[b*8+:8]=8'hff;
  wire [103:0] qdata;
  wire qpop=aligned_valid[b] && aligned_ready[b];
  banked_result_fifo unscaled(.clk(clk),.reset(reset || cancel),.push(take),
   .in_data({16'd0,8'hff,tag,in_data[(b*2+1)*64+:64]}),
   .out_valid(qvalid[b]),.out_ready(qpop),.out_data(qdata),.empty(qempty[b]),.error(qerror[b]));
  assign pair_bad[b]=aligned_valid[b] && qvalid[b] &&
   (aligned_tag[b*16+:16]!=qdata[79:64] || aligned_mask[b*8+:8]!=8'hff ||
    aligned_bias_overflow[b*8+:8]!=0 || aligned_range_error[b*8+:8]!=0);
  assign bypass_valid[b+2]=enable && aligned_valid[b] && qvalid[b] && !pair_bad[b];
  assign aligned_ready[b]=enable && qvalid[b] && bypass_ready[b+2] && !pair_bad[b];
  assign bypass_tag[(b+2)*16+:16]=qdata[79:64];assign bypass_mask[(b+2)*8+:8]=8'hff;
  for(genvar l=0;l<8;l=l+1)begin:LANE
   wire signed [7:0] scale_input=in_data[(b*2*8+l)*8+:8];
   wire signed [7:0] a=aligned_data[(b*8+l)*8+:8],c=qdata[l*8+:8];
   assign bypass_value[(b*8+l)*9+:9]={scale_input[7],scale_input};
   assign bypass_value[((b+2)*8+l)*9+:9]=$signed({a[7],a})+$signed({c[7],c});
  end
 end
 always @(posedge clk)begin
  if(reset || cancel)begin running<=0;done_valid<=0;error<=0;pixels<=0;rows<=0;groups<=0;lg<=0;next_index<=0;total<=0;accepted<=0;end
  else begin
   if(done_valid && done_ready)done_valid<=0;
   if(cmd_valid && cmd_ready)begin
    running<=1;error<=0;pixels<=cmd_pixels;rows<=cmd_rows;groups<=cmd_groups;next_index<=0;accepted<=0;
    lg<=cmd_groups==4 ? 2'd2 : cmd_groups==2 ? 2'd1 : 2'd0;
    total<=13'(((int'(cmd_pixels)+3)>>2)<<((cmd_groups==4 ? 2 : cmd_groups==2 ? 1 : 0)+2));
    if(cmd_pixels==0 || cmd_pixels>320 || cmd_rows==0 || cmd_rows>2 ||
       !(cmd_groups==1 || cmd_groups==2 || cmd_groups==4))begin error<=1;running<=0;done_valid<=1;end
   end
   if(pop)begin
    if(in_index!=next_index)error<=1;
    next_index<=next_index+1'b1;accepted<=0;
    if(next_index+1'b1==total)begin running<=0;done_valid<=1;end
   end else accepted<=accepted|(bypass_valid[1:0]&bypass_ready[1:0]);
   if((|qerror)||(|pair_bad))error<=1;
  end
 end
endmodule
