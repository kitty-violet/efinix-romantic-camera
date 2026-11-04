// Isolated A/B: preserve every public cycle of int8_dual_mac_lane_dspacc.
// A/B capture every clock using the DSP's independent *_REG_USE_CE=0.
// W still updates only for the previous accepted input (CE=valid_q0).
// No CASCOUT routing is used. A local clear also clears the hidden input
// registers, which cannot affect an output because both valid stages clear.
module int8_dual_mac_lane_dspacc (
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    clear,
    input  wire                    in_valid,
    input  wire signed [7:0]       activation0,
    input  wire signed [7:0]       weight0,
    input  wire signed [7:0]       activation1,
    input  wire signed [7:0]       weight1,
    output wire                    out_valid,
    output wire signed [31:0]      accumulator0,
    output wire signed [31:0]      accumulator1
);
    reg valid_q0,valid_q1;
    wire reset_acc=!rst_n || clear;
    wire [23:0] sum0,sum1;
    assign out_valid=valid_q1;
    assign accumulator0={{8{sum0[23]}},sum0};
    assign accumulator1={{8{sum1[23]}},sum1};
    always @(posedge clk)begin
        if(reset_acc)begin valid_q0<=0;valid_q1<=0;end
        else begin valid_q0<=in_valid;valid_q1<=valid_q0;end
    end
    EFX_DSP24 #(
        .A_REG(1'b1),.B_REG(1'b1),.P_REG(1'b0),.W_REG(1'b1),.O_REG(1'b0),
        .A_REG_USE_CE(1'b0),.B_REG_USE_CE(1'b0),.W_REG_USE_CE(1'b1),
        .A_REG_USE_RST(1'b1),.B_REG_USE_RST(1'b1),.W_REG_USE_RST(1'b1),
        .RST_SYNC(1'b1),.RST_POLARITY(1'b1),.SIGNED(1'b1),
        .M_SEL("P"),.N_SEL("W"),.W_SEL("X"),.CASCOUT_SEL("W")
    ) lane0 (
        .A(activation0),.B(weight0),.C(8'd0),.CASCIN(24'd0),.OP(2'd0),
        .SHIFT_ENA(1'b0),.CLK(clk),.CE(valid_q0),.RST(reset_acc),
        .O(sum0),.CASCOUT()
    );
    EFX_DSP24 #(
        .A_REG(1'b1),.B_REG(1'b1),.P_REG(1'b0),.W_REG(1'b1),.O_REG(1'b0),
        .A_REG_USE_CE(1'b0),.B_REG_USE_CE(1'b0),.W_REG_USE_CE(1'b1),
        .A_REG_USE_RST(1'b1),.B_REG_USE_RST(1'b1),.W_REG_USE_RST(1'b1),
        .RST_SYNC(1'b1),.RST_POLARITY(1'b1),.SIGNED(1'b1),
        .M_SEL("P"),.N_SEL("W"),.W_SEL("X"),.CASCOUT_SEL("W")
    ) lane1 (
        .A(activation1),.B(weight1),.C(8'd0),.CASCIN(24'd0),.OP(2'd0),
        .SHIFT_ENA(1'b0),.CLK(clk),.CE(valid_q0),.RST(reset_acc),
        .O(sum1),.CASCOUT()
    );
endmodule
