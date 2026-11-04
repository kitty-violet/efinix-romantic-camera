// Candidate contiguous-column kernel. depthwise remains fixed until drain
// completes; first two columns per row warm the window, no result is used.
// Structural arithmetic candidate: four spatial rows x eight channels.
// Shared input skew is built once per row, shared weight skew once per
// channel. Eight direct DSP cascade taps plus one exact fabric product.
// This exposes the fixed dot8/dot9 primitive; supply/RQ/partial sums external.
module fixed_window_dot8x32 #(parameter integer WEIGHT_PRE_SKEWED=0, SHARED_COLUMN_ACTIVATION=0, TAIL_SHIFT_048=0)(
 input wire clk,reset,depthwise,drain,in_warmup,input wire in_valid,output wire in_ready,
 input wire [383:0] columns,
 input wire [287:0] activation,input wire [575:0] weight,
 output wire out_valid,input wire out_ready,output wire [671:0] dots
);
 reg [8:0] valid;
 wire room=!out_valid || out_ready;
 // Accepted columns define time. Input gaps freeze the complete arithmetic
 // wavefront; the last output may retire independently. Explicit drain ends
 // a line/command. Never insert a bubble into the spatial column history.
 wire advance=room && (in_valid || drain);
 // Local fanout branches follow physical row/channel consumers. EFX_LUT4
 // identity drivers plus syn_keep retain both the driver and its loads.
 // This is combinational distribution; advance/valid/drain retain their cycle.
 (* syn_keep="true" *) wire [5:0] history_advance;
 (* syn_keep="true" *) wire [3:0] early_depthwise,tail_advance;
 (* syn_keep="true" *) wire weight_advance;
 (* syn_keep="true" *) wire [15:0] pair_advance;
 (* syn_keep="true" *) wire [31:0] lane_depthwise;
 for(genvar local_row=0;local_row<5;local_row=local_row+1)begin:HISTORY_CONTROL
  EFX_LUT4 #(.LUTMASK(16'haaaa)) enable_driver(.I0(advance),.I1(1'b0),.I2(1'b0),.I3(1'b0),.O(history_advance[local_row]));
 end
 for(genvar local_row=0;local_row<3;local_row=local_row+1)begin:ROW_CONTROL
  EFX_LUT4 #(.LUTMASK(16'haaaa)) mode_driver(.I0(depthwise),.I1(1'b0),.I2(1'b0),.I3(1'b0),.O(early_depthwise[local_row]));
  EFX_LUT4 #(.LUTMASK(16'haaaa)) tail_driver(.I0(advance),.I1(1'b0),.I2(1'b0),.I3(1'b0),.O(tail_advance[local_row]));
  for(genvar local_pair=0;local_pair<4;local_pair=local_pair+1)begin:PAIR
   EFX_LUT4 #(.LUTMASK(16'haaaa)) enable_driver(.I0(advance),.I1(1'b0),.I2(1'b0),.I3(1'b0),.O(pair_advance[local_row*4+local_pair]));
  end
  for(genvar local_channel=0;local_channel<8;local_channel=local_channel+1)begin:CHANNEL
   EFX_LUT4 #(.LUTMASK(16'haaaa)) mode_driver(.I0(depthwise),.I1(1'b0),.I2(1'b0),.I3(1'b0),.O(lane_depthwise[local_row*8+local_channel]));
  end
 end
 EFX_LUT4 #(.LUTMASK(16'haaaa)) weight_enable_driver(.I0(advance),.I1(1'b0),.I2(1'b0),.I3(1'b0),.O(weight_advance));
 assign in_ready=!reset && !drain && room;
 assign out_valid=!reset && valid[8];
 always @(posedge clk)begin
  if(reset)valid<=0;
  else if(advance)valid<={valid[7:0],in_valid && !drain && !in_warmup};
  else if(out_valid && out_ready)valid[8]<=0;
 end
 // With contiguous accepted columns, tap j=3*ky+kx needs skew
 // j+(2-kx)=3*ky+2. All three x taps share one delayed row value.
 // Only the rows consumed by each ky are instantiated; no nine-way router.
 // Share the two-step DW history and dense term skew in mutually exclusive
 // modes. Q7 is cascade-only in Ti60 hardware: all payload taps use Q.
 wire [63:0] dw[0:2][0:2];
 wire [63:0] a[0:2],b[0:7],delay2[0:2],delay5[1:3],delay8[2:4];
 for(genvar row=0;row<5;row=row+1)begin:INPUT_ROW
  for(genvar k=0;k<64;k=k+1)begin:BIT
   if(row<3)begin:EARLY
    localparam integer T=k/8;
    wire q;
    wire d=SHARED_COLUMN_ACTIVATION!=0 ? activation[(row*9+T)*8+k%8]:columns[row*64+k];
    wire [2:0] tap=SHARED_COLUMN_ACTIVATION!=0 && !early_depthwise[row] ? 3'(8-T):3'd6;
    EFX_SRL8 history(.D(d),.A(tap),.CLK(clk),.CE(history_advance[row]),.Q(q),.Q7());
    assign delay2[row][k]=~q;
   end
   if(row>=1 && row<=3)begin:MIDDLE
    wire q;EFX_SRL8 history(.D(columns[row*64+k]),.A(3'd3),.CLK(clk),.CE(history_advance[row]),.Q(q),.Q7());
    assign delay5[row][k]=~q;
   end
   if(row>=2)begin:LAST
    wire q;EFX_SRL8 history(.D(columns[row*64+k]),.A(3'd0),.CLK(clk),.CE(history_advance[row]),.Q(q),.Q7());
    assign delay8[row][k]=~q;
   end
  end
 end
 for(genvar row=0;row<3;row=row+1)begin:DW_TAPS
  assign dw[row][0]=delay2[row];assign dw[row][1]=delay5[row+1];assign dw[row][2]=delay8[row+2];
 end
 // synthesis translate_off
 always @(posedge clk)if(SHARED_COLUMN_ACTIVATION!=0 && !reset && depthwise && in_valid)
  for(integer row=0;row<3;row=row+1)for(integer t=0;t<8;t=t+1)
   if(activation[(row*9+t)*8+:8]!==columns[row*64+t*8+:8])$fatal(1,"SHARED_HISTORY_SOURCE_CONTRACT");
 // synthesis translate_on
  // Delay the shared operands before the ninth product, rather than
 // delaying32 independent16-bit products. The DW8-cycle tap already exists.
 wire [7:0] tail_dense[0:2],tail_weight[0:7];
 for(genvar r=0;r<3;r=r+1)begin:TAIL_A
  for(genvar k=0;k<8;k=k+1)begin:BIT
   wire q;EFX_SRL8 delay(.D(activation[(r*9+8)*8+k]),.A(3'd0),.CLK(clk),.CE(tail_advance[r]),.Q(q),.Q7());
   assign tail_dense[r][k]=~q;
  end
 end
 for(genvar c=0;c<8;c=c+1)begin:TAIL_B
  for(genvar k=0;k<8;k=k+1)begin:BIT
   if(WEIGHT_PRE_SKEWED!=0)assign tail_weight[c][k]=weight[(c*9+8)*8+k];
   else begin:DELAY
   wire q;EFX_SRL8 delay(.D(weight[(c*9+8)*8+k]),.A(3'd0),.CLK(clk),.CE(weight_advance),.Q(q),.Q7());
   assign tail_weight[c][k]=~q;
   end
  end
 end
 for(genvar t=0;t<8;t=t+1)begin:SKEW
  for(genvar r=0;r<3;r=r+1)begin:ROW
   for(genvar bitn=0;bitn<8;bitn=bitn+1)begin:BIT
    if(t==0)assign a[r][t*8+bitn]=activation[(r*9+t)*8+bitn];
    else if(SHARED_COLUMN_ACTIVATION!=0)assign a[r][t*8+bitn]=delay2[r][t*8+bitn];
    else begin
     wire q;EFX_SRL8 delay(.D(activation[(r*9+t)*8+bitn]),.A(3'(8-t)),.CLK(clk),.CE(history_advance[r]),.Q(q),.Q7());
     assign a[r][t*8+bitn]=~q;
    end
   end
  end
  for(genvar c=0;c<8;c=c+1)begin:CHANNEL
   for(genvar bitn=0;bitn<8;bitn=bitn+1)begin:BIT
    if(t==0 || WEIGHT_PRE_SKEWED!=0)assign b[c][t*8+bitn]=weight[(c*9+t)*8+bitn];
    else begin
     wire q;EFX_SRL8 delay(.D(weight[(c*9+t)*8+bitn]),.A(3'(8-t)),.CLK(clk),.CE(weight_advance),.Q(q),.Q7());
     assign b[c][t*8+bitn]=~q;
    end
   end
  end
 end wire [31:0] tail98_advance,tail98_mode,tail98_dense;
 wire [255:0] tail98_dw;wire [63:0] tail98_weight;
 wire [607:0] tail98_sum8,tail98_sum7,tail98_dot9;
 wire [767:0] tail99_cascade8;
 for(genvar tr=0;tr<3;tr=tr+1)begin:TAIL98_ROW
  assign tail98_dense[tr*8+:8]=~tail_dense[tr];
  assign tail98_dw[tr*64+:64]=~dw[tr][2];
  for(genvar tc=0;tc<8;tc=tc+1)begin:CHANNEL
   assign tail98_advance[tr*8+tc]=pair_advance[tr*4+tc/2];
   assign tail98_mode[tr*8+tc]=lane_depthwise[tr*8+tc];
   assign tail98_sum8[(tr*8+tc)*19+:19]=ROW[tr].PAIR[tc/2].sums[7][(tc%2)*24+:19];
   assign tail98_sum7[(tr*8+tc)*19+:19]=ROW[tr].PAIR[tc/2].sums[6][(tc%2)*24+:19];
   assign tail99_cascade8[(tr*8+tc)*24+:24]=ROW[tr].PAIR[tc/2].cascade[8][(tc%2)*24+:24];
  end
 end
 for(genvar tc=0;tc<8;tc=tc+1)assign tail98_weight[tc*8+:8]=tail_weight[tc];
 assign tail98_dense[24+:8]=0;assign tail98_dw[192+:64]=0;
 assign tail98_advance[24+:8]=0;assign tail98_mode[24+:8]=0;
 assign tail98_sum7[456+:152]=0;assign tail98_sum8[456+:152]=0;assign tail99_cascade8[576+:192]=0;
 assign dots[504+:168]=dots[336+:168];
 generic_b1_tail94 tail98(.sum7(tail98_sum7),.cascade8(tail99_cascade8),.clk(clk),.advance(tail98_advance),.depthwise(tail98_mode),
  .dense_q(tail98_dense),.dw_q(tail98_dw),.weight(tail98_weight),.sum8(tail98_sum8),.dot9(tail98_dot9));
 // synthesis translate_off
 initial if(TAIL_SHIFT_048!=0)$fatal(1,"98 requires complete B1 tail coefficients");
 // synthesis translate_on

 for(genvar r=0;r<3;r=r+1)begin:ROW
  for(genvar k=0;k<4;k=k+1)begin:PAIR
   wire [47:0] cascade[0:8],sums[0:7];assign cascade[0]=0;
   for(genvar t=0;t<8;t=t+1)begin:TAP
    wire [7:0] a0=lane_depthwise[r*8+k*2] ? dw[r][t/3][k*16+:8]:a[r][t*8+:8];
    wire [7:0] a1=lane_depthwise[r*8+k*2+1] ? dw[r][t/3][k*16+8+:8]:a[r][t*8+:8];
    wire [7:0] b0=b[k*2][t*8+:8],b1=b[k*2+1][t*8+:8];
    EFX_DSP48 #(.MODE("DUAL"),.A_REG(1'b1),.B_REG(1'b1),.P_REG(1'b0),.W_REG(1'b1),.O_REG(1'b0),
     .A_REG_USE_CE(1'b1),.B_REG_USE_CE(1'b1),.W_REG_USE_CE(1'b1),
     .A_REG_USE_RST(1'b0),.B_REG_USE_RST(1'b0),.W_REG_USE_RST(1'b0),
     .RST_SYNC(1'b1),.SIGNED(1'b1),.M_SEL("P"),.N_SEL(t==0 ? "CONST0":(t==7 && k!=0) ? "C":"CASCIN"),
     .W_SEL("X"),.CASCOUT_SEL("W")) mac(
     .A({{3{a1[7]}},a1,a0}),.B({{2{b1[7]}},b1,b0}),.C((t==7 && k!=0) ? {sums[6][24+:10],sums[6][0+:8]}:18'd0),.CASCIN((t==7 && k!=0) ? 48'd0:cascade[t]),.OP(2'd0),.SHIFT_ENA(1'b0),
     .CLK(clk),.CE(pair_advance[r*4+k]),.RST(1'b0),.O(sums[t]),.CASCOUT(cascade[t+1]),.OVFL());
   end
  end
  for(genvar c=0;c<8;c=c+1)begin:CHANNEL
   wire signed [18:0] dot9=tail98_dot9[(r*8+c)*19+:19];
   assign dots[(r*8+c)*21+:21]={{2{dot9[18]}},dot9};
  end
 end
endmodule

// Explicit dual8-bit tail multiplication. The compiled B1 connects each
// adjacent pair to exactly the same enable (checked against native97 netlist).
module generic_b1_tail94(
 input wire clk, input wire [31:0] advance,depthwise,
 input wire [31:0] dense_q,input wire [255:0] dw_q,
 input wire [63:0] weight,input wire [607:0] sum8,
 input wire [767:0] cascade8,input wire [607:0] sum7,
 output wire [607:0] dot9
);
 assign dot9[456+:152]=0;
 for(genvar r=0;r<3;r=r+1)begin:ROW
  for(genvar k=0;k<4;k=k+1)begin:PAIR
   localparam integer L=r*8+k*2;
   wire [7:0] a0=depthwise[L] ? ~dw_q[L*8+:8] : ~dense_q[r*8+:8];
   wire [7:0] a1=depthwise[L+1] ? ~dw_q[(L+1)*8+:8] : ~dense_q[r*8+:8];
   wire [7:0] b0=weight[k*16+:8],b1=weight[k*16+8+:8];
   wire [47:0] products;
   EFX_DSP48 #(.MODE("DUAL"),.A_REG(1),.B_REG(1),.W_REG(0),.O_REG(0),.P_REG(0),
    .A_REG_USE_CE(1),.B_REG_USE_CE(1),.A_REG_USE_RST(0),.B_REG_USE_RST(0),
    .SIGNED(1),.M_SEL("P"),.N_SEL("CASCIN"),.W_SEL("X"),.CASCOUT_SEL("W")) tail(
    .A({{3{a1[7]}},a1,a0}),.B({{2{b1[7]}},b1,b0}),.C(18'd0),.CASCIN({cascade8[(L+1)*24+:24],cascade8[L*24+:24]}),.OP(2'b00),
    .SHIFT_ENA(1'b0),.CLK(clk),.CE(advance[L]),.RST(1'b0),.O(products),.CASCOUT(),.OVFL());
   // Registered previous8-tap W outputs feed the dedicated cascade input.
   // Tail operand input registers are unchanged; no new output register.
   if(k==0)begin:FULL_CASCADE
    assign dot9[L*19+:19]=products[18:0];
    assign dot9[(L+1)*19+:19]=products[42:24];
   end else begin:SPLIT_7_2
    // The deployed 222 placement splits seven taps plus two. C is only
    // 8/10 bits in DUAL mode: carry the remaining subtotal bits across the
    // original eighth W register, then reconstruct exactly (no extra cycle).
    reg [37:0] high_q;
    always @(posedge clk)if(advance[L])high_q<={sum7[(L+1)*19+:19],sum7[L*19+:19]};
    for(genvar half=0;half<2;half=half+1)begin:HALF
     localparam integer K=half ? 10:8,W=19-K,N=L+half;
     wire [W:0] carry;
     EFX_ADD entry(.I0(high_q[half*19+K-1]),.I1(1'b1),.CI(1'b0),.O(),.CO(carry[0]));
     assign dot9[N*19+:K]=products[half*24+:K];
     for(genvar j=0;j<W;j=j+1)begin:HIGH
      EFX_ADD add(.I0(high_q[half*19+K+j]),.I1(products[half*24+K+j]),.CI(carry[j]),.O(dot9[N*19+K+j]),.CO(carry[j+1]));
     end
    end
   end
   // synthesis translate_off
   always @(posedge clk) if(advance[L]!==advance[L+1])
     $fatal(1,"DUAL tail requires structurally identical pair enables");
   // synthesis translate_on
  end
 end
endmodule
