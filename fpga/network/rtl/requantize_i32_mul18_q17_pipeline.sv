// Exact INT32 * signed18 Q17, unchanged four-edge public latency.
// Move operand/correction registers into the existing DSP, then use its
// output register to retain the original pipeline depth. No extra DSP.
module requantize_i32_mul18_q17_pipeline (
 input wire clk,rst_n,in_valid,
 input wire signed [31:0] value,multiplier_field,
 input wire relu_enable,
 output reg out_valid,
 output reg signed [7:0] result,
 output reg multiplier_range_error
);
 wire fits19=value[31:19]=={13{value[18]}};
 wire fits25=value[31:25]=={7{value[24]}};
 wire fits7=multiplier_field[17:7]=={11{multiplier_field[6]}};
 wire split=!fits19 && fits25 && fits7;
 wire saturated=!fits19 && !split && multiplier_field[17:0]!=0;
 wire signed [18:0] operand_a=split ? $signed(value[24:6]):$signed(value[18:0]);
 wire signed [17:0] operand_b=split ? {multiplier_field[11:0],6'd0}:multiplier_field[17:0];
 wire [5:0] low_bits=split ? value[5:0]:6'd0;
 (* syn_use_dsp="no" *) wire signed [13:0] correction=$signed({1'b0,low_bits})*$signed(multiplier_field[6:0]);
 reg sat0,neg0,relu0,err0,v0;
 reg sat1,neg1,relu1,err1,v1;
 reg sat2,neg2,relu2,err2,v2;
 wire signed [47:0] product2;
 EFX_DSP48 #(.A_REG(1),.B_REG(1),.C_REG(1),.P_REG(0),.W_REG(1),.O_REG(1),
  .RST_SYNC(1),.SIGNED(1),.P_EXT("ALIGN_RIGHT"),.C_EXT("ALIGN_RIGHT"),
  .M_SEL("P"),.N_SEL("C"),.W_SEL("X"),.CASCOUT_SEL("W")) multiply_add (
  .A(operand_a),.B(operand_b),.C({{4{correction[13]}},correction}),.CASCIN(48'd0),
  .OP(2'b00),.SHIFT_ENA(1'b0),.CLK(clk),.CE(1'b1),.RST(!rst_n),
  .O(product2),.CASCOUT(),.OVFL());
 function automatic signed [7:0] rounded;
  input signed [49:0] p;
  input relu;
  reg increment;
  reg [7:0] rounded_byte;
  begin
   increment=p[16] && (!p[49] || (|p[15:0]));
   rounded_byte=p[24:17]+{7'd0,increment};
   if(p[49] && relu)rounded=0;
   else if(!p[49])rounded=(|p[48:24]) || rounded_byte[7] ? 8'sd127:$signed(rounded_byte);
   else rounded=!(&p[48:24]) ? -8'sd128:$signed(rounded_byte);
  end
 endfunction
 always @(posedge clk)begin
  sat0<=saturated;neg0<=value[31]^multiplier_field[17];
  relu0<=relu_enable;err0<=multiplier_field[31:18]!={14{multiplier_field[17]}};
  sat1<=sat0;neg1<=neg0;relu1<=relu0;err1<=err0;
  sat2<=sat1;neg2<=neg1;relu2<=relu1;err2<=err1;
 end
 always @(posedge clk)begin
  if(!rst_n)begin
   v0<=0;v1<=0;v2<=0;out_valid<=0;result<=0;multiplier_range_error<=0;
  end else begin
   v0<=in_valid;v1<=v0;v2<=v1;out_valid<=v2;
   if(v2)begin
    result<=sat2 ? ((neg2 && relu2) ? 8'sd0:neg2 ? -8'sd128:8'sd127):rounded({{2{product2[47]}},product2},relu2);
    multiplier_range_error<=err2;
   end
  end
 end
endmodule
