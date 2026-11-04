// Four signed DOT_WIDTH values plus an unsigned low20 bias fit DOT_WIDTH+3
// bits for DOT_WIDTH=19..21. The real nine-product array produces signed19.
// Preserve the public 24-bit response and reject any out-of-contract input.
module fixed_partial32 #(parameter integer DOT_WIDTH=21)(
 input wire clk,reset,cancel,input wire in_valid,output wire in_ready,
 input wire in_first,in_last,input wire [2:0] in_parts,input wire [8:0] in_x,
 input wire [671:0] in_dots,input wire [255:0] bias,
 output wire out_valid,input wire out_ready,output wire [767:0] out_sums,output wire [8:0] out_x,
 output wire busy,output reg error
);
 localparam integer ACC_WIDTH=DOT_WIDTH+3;
 reg opened,full;reg [2:0] parts,count;reg [8:0] x;reg [24*ACC_WIDTH-1:0] sums;
 wire active=!reset && !cancel && !error;
 assign in_ready=active && (!full || out_ready);
 assign out_valid=active && full;assign out_x=x;assign busy=opened || full;
 wire fire=in_valid && in_ready;
 wire bad_format=!opened ? (!in_first || in_parts<1 || in_parts>4 || in_last!=(in_parts==1)):
  (in_first || in_x!=x || in_parts!=parts || in_last!=(count+3'd1==parts));
 wire [31:0] lane_bad;
 assign lane_bad[24+:8]=0;assign out_sums[576+:192]=out_sums[384+:192];
 wire bad=bad_format || (|lane_bad);
 for(genvar lane=0;lane<24;lane=lane+1)begin:LANE
  wire signed [DOT_WIDTH-1:0] dot=in_dots[lane*21+:DOT_WIDTH];
  if(DOT_WIDTH<21)assign lane_bad[lane]=in_dots[lane*21+DOT_WIDTH+:21-DOT_WIDTH]!={(21-DOT_WIDTH){dot[DOT_WIDTH-1]}};
  else assign lane_bad[lane]=1'b0;
  assign out_sums[lane*24+:24]={{(24-ACC_WIDTH){sums[lane*ACC_WIDTH+ACC_WIDTH-1]}},sums[lane*ACC_WIDTH+:ACC_WIDTH]};
  wire signed [31:0] b=bias[(lane%8)*32+:32];
  wire signed [ACC_WIDTH-1:0] prior=in_first ? {{(ACC_WIDTH-20){1'b0}},b[19:0]}:sums[lane*ACC_WIDTH+:ACC_WIDTH];
  always @(posedge clk)begin
   // Valid/owner state resets; a first phase overwrites every payload bit.
   // Resetting the unused payload adds a redundant wide reset/mux network.
   if(fire && !bad)sums[lane*ACC_WIDTH+:ACC_WIDTH]<=prior+{{3{dot[DOT_WIDTH-1]}},dot};
  end
 end
 always @(posedge clk)begin
  if(reset || cancel)begin opened<=0;full<=0;parts<=0;count<=0;x<=0;error<=0;end
  else if(!error)begin
   if(out_valid && out_ready)full<=0;
   if(fire)begin
    if(bad)error<=1;
    else begin
     if(in_first)begin x<=in_x;parts<=in_parts;count<=1;end else count<=count+3'd1;
     opened<=!in_last;if(in_last)full<=1;
    end
   end
  end
 end
endmodule
