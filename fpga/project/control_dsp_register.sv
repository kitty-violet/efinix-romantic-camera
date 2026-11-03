// Documented NORMAL-mode P/C path: A*2^17+C concatenates 19+17 bits.
// Only O is registered. Cascade ports are unused; no extra clock or latency.
module control_dsp_register #(parameter WIDTH=32,RST_SYNC=1)(
 input wire clk,reset,enable,input wire[WIDTH-1:0]data,output wire[WIDTH-1:0]q
);
 // Isolate APB read write-data X from the address/sequence/control field.
 localparam GAP=(WIDTH>36)?4:0;
 localparam BLOCKS=(WIDTH+GAP+35)/36;
 wire[36*BLOCKS-1:0]padded,result;
 generate if(GAP)assign padded={{(36*BLOCKS-WIDTH-GAP){1'b0}},data[WIDTH-1:32],4'd0,data[31:0]};
 else assign padded={{(36*BLOCKS-WIDTH){1'b0}},data};endgenerate
 genvar k;
 generate for(k=0;k<BLOCKS;k=k+1)begin:bank
  wire[35:0]word_in=padded[k*36+:36];wire[47:0]word_out;
  EFX_DSP48 #(.MODE("NORMAL"),.A_REG(0),.B_REG(0),.C_REG(0),.P_REG(0),
   .OP_REG(0),.W_REG(0),.O_REG(1),.SHIFTER(0),.SIGNED(0),
   .M_SEL("P"),.N_SEL("C"),.W_SEL("X"),.RST_SYNC(RST_SYNC)) dsp(
   .A(word_in[35:17]),.B(18'h20000),.C({1'b0,word_in[16:0]}),
   .CASCIN(48'd0),.OP(2'b00),.SHIFT_ENA(1'b0),.CLK(clk),.CE(enable),.RST(reset),
   .O(word_out),.CASCOUT(),.OVFL());
  assign result[k*36+:36]=word_out[35:0];
 end endgenerate
 generate if(GAP)assign q={result[WIDTH+GAP-1:36],result[31:0]};
 else assign q=result[WIDTH-1:0];endgenerate
endmodule

// Internal W feedback consumes no ordinary fabric adder. Modulo truncation
// preserves every original counter width, including wraparound.
module control_dsp_counter #(parameter WIDTH=32,RST_SYNC=1)(
 input wire clk,reset,enable,output wire[WIDTH-1:0]q
);
 wire[47:0]value;
 EFX_DSP48 #(.MODE("NORMAL"),.A_REG(0),.B_REG(0),.C_REG(0),.P_REG(0),
  .OP_REG(0),.W_REG(1),.O_REG(0),.SHIFTER(0),.SIGNED(0),
  .M_SEL("C"),.N_SEL("W"),.W_SEL("X"),.RST_SYNC(RST_SYNC)) dsp(
  .A(19'd0),.B(18'd0),.C(18'd1),.CASCIN(48'd0),.OP(2'b00),
  .SHIFT_ENA(1'b0),.CLK(clk),.CE(enable),.RST(reset),.O(value),.CASCOUT(),.OVFL());
 assign q=value[WIDTH-1:0];
endmodule
