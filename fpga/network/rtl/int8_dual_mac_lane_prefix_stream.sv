// Fixed scheduled bank: reserve at accepted LAST, consume the DSP prefix
// at its first valid tail edge. No private wide result mailbox is required.
// DSP primitive and free-running feedback remain the qualified implementation.
module int8_dual_mac_lane_prefix_stream #(parameter integer EXTERNAL_PREVIOUS=0) (
 input wire clk,rst_n,clear,consume,epoch_clear,
 output wire [41:0] previous_pair,
 input wire in_valid,first_term,last_term,
 input wire signed [7:0] activation0,weight0,activation1,weight1,
 output wire dot_valid,output wire signed [31:0] dot0,dot1,
 output reg protocol_error
);
 wire accumulator_valid;wire signed [31:0] prefix0,prefix1;
 reg [1:0] valid_pipe,last_pipe;
 generate if(EXTERNAL_PREVIOUS==0)begin:PRIVATE_HISTORY
  reg [20:0] previous0,previous1;
  assign previous_pair={previous1,previous0};
  always @(posedge clk)begin
   if(!rst_n || clear)begin previous0<=0;previous1<=0;end
   else if(consume)begin previous0<=prefix0[20:0];previous1<=prefix1[20:0];end
  end
 end else begin:BANK_HISTORY
  assign previous_pair=42'd0;
 end endgenerate
 assign dot_valid=rst_n && !clear && !protocol_error && accumulator_valid && last_pipe[1];
 assign dot0={{11{prefix0[20]}},prefix0[20:0]};
 assign dot1={{11{prefix1[20]}},prefix1[20:0]};
 int8_dual_mac_lane_dspacc running(.clk(clk),.rst_n(rst_n),.clear(clear || (EXTERNAL_PREVIOUS!=0 && epoch_clear)),.in_valid(in_valid),
  .activation0(activation0),.weight0(weight0),.activation1(activation1),.weight1(weight1),
  .out_valid(accumulator_valid),.accumulator0(prefix0),.accumulator1(prefix1));
 always @(posedge clk)begin
  if(!rst_n || clear || (EXTERNAL_PREVIOUS!=0 && epoch_clear))begin
   valid_pipe<=0;last_pipe<=0;protocol_error<=0;
  end else begin
   valid_pipe<={valid_pipe[0],in_valid};last_pipe<={last_pipe[0],in_valid && last_term};
   if(accumulator_valid!=valid_pipe[1] || consume!=dot_valid)protocol_error<=1;
  end
 end
endmodule
