// Expand four packed RGB results into up to eight independently acknowledged
// writes. A pair stays at the original FIFO head until both pixels are taken.
module result_pair_expand4(
 input wire clk,reset,cancel,pair_mode,input wire [8:0] pixels,
 input wire [3:0] in_valid,output wire [3:0] in_ready,input wire [255:0] in_data,input wire [63:0] in_tag,
 input wire [31:0] in_mask,in_bias_overflow,in_range_error,
 output wire [7:0] out_valid,input wire [7:0] out_ready,output wire [511:0] out_data,output wire [127:0] out_tag,
 output wire [63:0] out_mask,out_bias_overflow,out_range_error,output wire busy,output reg error
);
 reg [7:0] taken;wire active=!reset && !cancel && !error;wire [3:0] bad;
 assign busy=|taken;
 for(genvar b=0;b<4;b=b+1)begin:PAIR
  wire [15:0] tag=in_tag[b*16+:16];wire [9:0] x={tag[10:2],1'b0};
  wire second=pair_mode && x+10'd1<{1'b0,pixels};
  assign bad[b]=in_valid[b] && pair_mode && (tag[15:13]!=0 || tag[1:0]!=0 || x>={1'b0,pixels} ||
   in_mask[b*8+:8]!=8'h3f || (|in_bias_overflow[b*8+:8]) || (|in_range_error[b*8+:8]));
  assign out_valid[b*2]=active && in_valid[b] && !bad[b] && !taken[b*2];
  assign out_valid[b*2+1]=active && in_valid[b] && !bad[b] && second && !taken[b*2+1];
  assign in_ready[b]=active && !bad[b] && (taken[b*2] || out_ready[b*2]) && (!second || taken[b*2+1] || out_ready[b*2+1]);
  assign out_data[b*128+:128]=pair_mode ? {40'd0,in_data[b*64+24+:24],40'd0,in_data[b*64+:24]} : {64'd0,in_data[b*64+:64]};
  assign out_tag[b*32+:32]=pair_mode ? {{3'd0,tag[12:11],9'(x+10'd1),2'd0},{3'd0,tag[12:11],x[8:0],2'd0}} : {16'd0,tag};
  assign out_mask[b*16+:16]=pair_mode ? 16'h0707 : {8'd0,in_mask[b*8+:8]};
  assign out_bias_overflow[b*16+:16]=pair_mode ? 16'd0 : {8'd0,in_bias_overflow[b*8+:8]};
  assign out_range_error[b*16+:16]=pair_mode ? 16'd0 : {8'd0,in_range_error[b*8+:8]};
  always @(posedge clk)begin
   if(reset || cancel)taken[b*2+:2]<=0;
   else if(in_valid[b] && in_ready[b])taken[b*2+:2]<=0;
   else taken[b*2+:2]<=taken[b*2+:2]|(out_valid[b*2+:2]&out_ready[b*2+:2]);
  end
 end
 always @(posedge clk)begin if(reset || cancel)error<=0;else if(|bad)error<=1;end
endmodule
