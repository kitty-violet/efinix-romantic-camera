// Shared nine-word routing for mutually exclusive stride and even-row DW modes.
// Candidate: payload delay registers keep their contents across reset/cancel.
// Ownership, indices and valid state still reset; no payload may be consumed
// before its corresponding accepted input/weight has filled that delay tap.
// Single mode-exclusive history/cache/weight pipeline for the 32 MAC fronts.
// Every phase advances only after all participating owners accept its token.
module unified_column_dispatch32(
 input wire clk,reset,cancel,input wire cmd_valid,output wire cmd_ready,
 input wire cmd_pair,cmd_rgb,cmd_mode_pw,cmd_stem,
 input wire [8:0] cmd_pixels,input wire [2:0] cmd_rows,cmd_in_groups,cmd_out_groups,
 input wire [1:0] cmd_group,input wire [7:0] cmd_mask,
 input wire in_valid,output wire in_ready,input wire [767:0] in_data,input wire [12:0] in_index,input wire in_warmup,
 input wire weight_valid,output wire weight_ready,input wire [63:0] weight_data,input wire [12:0] weight_index,
 output wire done_valid,input wire done_ready,output wire busy,output reg error,
 output wire [31:0] pool_valid,input wire [31:0] pool_ready,output wire [31:0] pool_first,pool_last,
 output wire [191:0] pool_terms,output wire [2047:0] pool_data,pool_weights,
 output wire [63:0] pool_group,output wire [511:0] pool_tag,output wire [255:0] pool_mask,
 output wire phase_advance,output reg [31:0] cycles,wait_input_cycles,wait_pool_cycles
);
 reg running,done,paired,rgb,pw_mode,stem,warming;
 reg [8:0] pixels,batch,batches,base;reg [2:0] rows;reg [1:0] group_q,log_cin,subphase;
 reg [6:0] phase,phase_mask;reg [4:0] term_mask;reg [5:0] terms;reg [7:0] mask;
 reg [12:0] next_column,next_weight;reg [31:0] owners,acked;
 wire [767:0] history[0:6];reg [383:0] left_column;wire [63:0] weights[0:20];
 wire active=running && !reset && !cancel && !error;
 wire [31:0] can_step;wire tokens=in_valid && weight_valid;
 wire ordinary_dw=!paired && !pw_mode;
 wire use_stride=paired && !rgb,use_dw=ordinary_dw && !stem;
 wire use_repeat=(!paired && (pw_mode || stem)) || rgb;
 assign cmd_ready=!reset && !cancel && !running && !done;
 assign done_valid=done && !reset && !cancel;assign busy=running||done;
 assign phase_advance=active && !warming && tokens && (&can_step);
 wire column_advance=phase_advance && (!ordinary_dw || !stem || subphase==2);
 assign in_ready=active && (warming || (weight_valid && (&can_step) && (!ordinary_dw || (in_valid && (!stem || subphase==2)))));
 assign weight_ready=ordinary_dw ? phase_advance:active && in_valid && (&can_step);
 wire input_fire=in_valid && in_ready;
 // Independent fixed taps deliberately trade replicated SRL bits for fewer
 // fabric registers. All original acceptance enables and tap latencies remain.
 reg [767:0] first_column;
 always @(posedge clk)if(input_fire)first_column<=in_data;
 assign history[0]=first_column;
 for(genvar t=1;t<7;t=t+1)begin:HISTORY_TAP
  for(genvar b=0;b<768;b=b+1)begin:BIT
   localparam integer WORD=b/64;
   localparam USED=(t==2 && WORD>=1 && WORD<=4) ||
     (t==3 && (WORD==1 || WORD==3 || WORD==7 || WORD==9)) ||
     (t==5 && WORD>=2 && WORD<=5) ||
     (t==6 && (WORD==2 || WORD==4 || WORD==8 || WORD==10));
   if(USED)begin:LIVE
    wire q;
    EFX_SRL8 tap(.D(first_column[b]),.A(3'(8-t)),.CLK(clk),.CE(column_advance),.Q(q),.Q7());
    assign history[t][b]=~q;
   end else assign history[t][b]=0;
  end
 end
 for(genvar t=0;t<21;t=t+1)begin:WEIGHT_TAP
  if(t<9 || t==11 || t==14 || t==17 || t==20)begin:USED
   localparam integer BASE=t<8 ? -1:t<15 ? 7:14;
   localparam integer DELAY=t-BASE;
   for(genvar b=0;b<64;b=b+1)begin:BIT
    wire q;
    EFX_SRL8 tap(.D(BASE<0 ? weight_data[b]:weights[BASE<0 ? 0:BASE][b]),
      .A(3'(8-DELAY)),.CLK(clk),.CE(phase_advance),.Q(q),.Q7());
    assign weights[t][b]=~q;
   end
  end else assign weights[t]=0;
 end
 wire [8:0] units=rgb ? ((pixels>>1)+{8'd0,pixels[0]}):pixels;
 wire [11:0] batch_x={batch,3'd0};
 // One nine-word cyclic permutation per stride row replaces nine independent
 // nine-way selectors. Only four binary rotation stages are required.
 wire [63:0] stride_rotated[0:1][0:8];
 for(genvar rr=0;rr<2;rr=rr+1)begin:STRIDE_ROTATE
  wire [575:0] stage[0:4];
  assign stage[2]=0;assign stage[3]=0;
  wire [575:0] taps={history[6][(rr*2+2)*64+:64],history[6][(rr*2+8)*64+:64],
   history[5][(rr*2+2)*64+:64],history[3][(rr*2+1)*64+:64],
   history[3][(rr*2+7)*64+:64],history[2][(rr*2+1)*64+:64],
   history[0][(rr*2)*64+:64],history[0][(rr*2+6)*64+:64],in_data[(rr*2)*64+:64]};
  // DW owner P consumes source floor(((phase-P) mod 9)/3).
  // Repeating each of its three row sources three times embeds that selection
  // in the very same cyclic router used by stride. No time multiplexing or
  // extra phase is introduced; both modes retain their original parallelism.
  wire [63:0] dw_top=in_data[(rr*2)*64+:64];
  wire [63:0] dw_mid=history[2][(rr*2+1)*64+:64];
  wire [63:0] dw_bottom=history[5][(rr*2+2)*64+:64];
  wire [575:0] dw_taps={{3{dw_bottom}},{3{dw_mid}},{3{dw_top}}};
  assign stage[0]=use_stride ? taps:dw_taps;
  // Phase has exactly nine legal values. Factor its cyclic permutation
  // into a row rotation and a column rotation, exposing the legal domain.
  wire [1:0] thirds=phase<3 ? 2'd0:phase<6 ? 2'd1:2'd2;
  wire rem_one=phase==1 || phase==4 || phase==7;
  wire rem_two=phase==2 || phase==5 || phase==8;
  for(genvar k=0;k<9;k=k+1)begin:TERNARY
   assign stage[1][k*64+:64]=thirds==0 ? stage[0][k*64+:64] :
    thirds==1 ? stage[0][((k+3)%9)*64+:64]:stage[0][((k+6)%9)*64+:64];
   assign stage[4][k*64+:64]=rem_one ? stage[1][((k+1)%9)*64+:64] :
    rem_two ? stage[1][((k+2)%9)*64+:64]:stage[1][k*64+:64];
  end
  for(genvar k=0;k<9;k=k+1)begin:OUT
   assign stride_rotated[rr][k]=stage[4][((9-k)%9)*64+:64];
  end
 end
 // PW/RGB accept an input exactly on each phase_advance. Their byte-j
 // delay therefore aliases column history[j-1], already maintained for DW.
 // Reuse only taps already exposed by DW/stride. Retain compact SRLs where
 // adding another history tap would otherwise force extra physical storage.
 // All modes own this history exclusively; valid owners hide warmup contents.
 wire [63:0] diagonal_a[0:3],diagonal_b[0:3];
 for(genvar r=0;r<4;r=r+1)begin:DIAGONAL_ROW
  for(genvar j=0;j<8;j=j+1)begin:BYTE
   if(j==0)begin
    assign diagonal_a[r][j*8+:8]=in_data[r*64+j*8+:8];
    assign diagonal_b[r][j*8+:8]=in_data[(r+6)*64+j*8+:8];
   end else begin
    localparam ALIAS_A=j==1 || (r==2 && (j==6 || j==7)) ||
     ((r==1 || r==3) && (j==3 || j==4));
    localparam ALIAS_B=((r==0 || r==2) && j==1) ||
     ((r==1 || r==3) && j==4) || (r==2 && j==7);
    if(ALIAS_A)assign diagonal_a[r][j*8+:8]=history[j-1][r*64+j*8+:8];
    else for(genvar b=0;b<8;b=b+1)begin:DELAY_A
     wire q;
     EFX_SRL8 delay(.D(in_data[r*64+j*8+b]),.A(3'(8-j)),.CLK(clk),.CE(phase_advance && (pw_mode || rgb)),.Q(q),.Q7());
     assign diagonal_a[r][j*8+b]=~q;
    end
    if(ALIAS_B)assign diagonal_b[r][j*8+:8]=history[j-1][(r+6)*64+j*8+:8];
    else for(genvar b=0;b<8;b=b+1)begin:DELAY_B
     wire q;
     EFX_SRL8 delay(.D(in_data[(r+6)*64+j*8+b]),.A(3'(8-j)),.CLK(clk),.CE(phase_advance && (pw_mode || rgb)),.Q(q),.Q7());
     assign diagonal_b[r][j*8+b]=~q;
    end
   end
  end
 end
 // The owner selects byte (phase-owner) mod 8. Share three rotation stages
 // across all eight owners instead of eight independent variable byte muxes.
 wire [7:0] scalar_a[0:3][0:7],scalar_b[0:3][0:7];
 for(genvar rr=0;rr<4;rr=rr+1)begin:BYTE_ROTATE
  wire [63:0] sa[0:3],sb[0:3];
  assign sa[0]=diagonal_a[rr];assign sb[0]=diagonal_b[rr];
  for(genvar ss=0;ss<3;ss=ss+1)begin:STAGE
   for(genvar k=0;k<8;k=k+1)begin:BYTE
    assign sa[ss+1][k*8+:8]=phase[ss] ? sa[ss][((k+(1<<ss))%8)*8+:8]:sa[ss][k*8+:8];
    assign sb[ss+1][k*8+:8]=phase[ss] ? sb[ss][((k+(1<<ss))%8)*8+:8]:sb[ss][k*8+:8];
   end
  end
  for(genvar k=0;k<8;k=k+1)begin:OUT
   assign scalar_a[rr][k]=sa[3][((8-k)%8)*8+:8];
   assign scalar_b[rr][k]=sb[3][((8-k)%8)*8+:8];
  end
 end
 // Three DW row sources are rotated once per row. Owners with the same
 // P/3 share that rotation; P%3 only selects across one phase boundary.
 wire [63:0] dw_rotation[0:3][0:2];
 wire dw_rem0=phase==0 || phase==3 || phase==6;
 wire dw_rem_lt2=phase!=2 && phase!=5 && phase!=8;
 for(genvar r=0;r<4;r=r+1)if(r%2==1)begin:DW_ROTATE
  wire [63:0] top_word=in_data[r*64+:64];
  wire [63:0] mid_word=history[2][(r+1)*64+:64];
  wire [63:0] bottom_word=history[5][(r+2)*64+:64];
  assign dw_rotation[r][0]=phase<3 ? top_word:phase<6 ? mid_word:bottom_word;
  assign dw_rotation[r][1]=phase<3 ? bottom_word:phase<6 ? top_word:mid_word;
  assign dw_rotation[r][2]=phase<3 ? mid_word:phase<6 ? bottom_word:top_word;
 end
 for(genvar o=0;o<32;o=o+1)begin:OWNER
  localparam integer R=o/8,P=o%8,DR=o/16,DP=o%16;
  wire [6:0] pw_age=(phase-7'(P)) & phase_mask;
  wire [4:0] pw_term={pw_age[4:3]&term_mask[4:3],pw_age[2:0]};
  wire [3:0] dw_age=phase[3:0]>=4'(P) ? phase[3:0]-4'(P):phase[3:0]+4'd9-4'(P);
  wire [3:0] pair_age=rgb ? {1'b0,3'(phase[3:0]-4'(P))} : phase[3:0]>=4'(DP) ? phase[3:0]-4'(DP):phase[3:0]+4'd9-4'(DP);
  wire [9:0] pair_x={1'b0,base}+(rgb ? 10'(P):10'(DP));
  wire pair_legal=rgb ? 3'(R)<rows:(DP<9 && 3'(DR)<rows);
  wire launch=paired ? pair_legal && pair_x<{1'b0,units} : batch<batches && 3'(R)<rows && batch_x+12'(P)<{3'd0,pixels};
  wire first=paired ? pair_age==0 : pw_mode ? pw_term==0:dw_age==0;
  wire last=paired ? pair_age==(rgb ? 4'd7:4'd8) : pw_mode ? pw_term==term_mask:dw_age==8;
  wire need=first ? launch:owners[o];
  assign pool_valid[o]=active && !warming && tokens && need && !acked[o];
  assign can_step[o]=!need || acked[o] || pool_ready[o];
  assign pool_first[o]=first && (!ordinary_dw || !stem || subphase==0);
  assign pool_last[o]=last && (!ordinary_dw || !stem || subphase==2);
  assign pool_terms[o*6+:6]=paired ? (rgb ? 6'd8:6'd9) : pw_mode ? terms:stem ? 6'd27:6'd9;
  wire [1:0] output_group=paired ? (rgb ? 2'd0:group_q) : pw_mode ? 2'((pw_age>>3)>>log_cin):group_q;
  assign pool_group[o*2+:2]=output_group;
  assign pool_mask[o*8+:8]=paired ? (rgb ? 8'h3f:8'hff) : pw_mode ? mask:8'hff;
  assign pool_tag[o*16+:16]=paired ? {3'd0,(rgb ? 2'(R):2'(DR)),pair_x[8:0],output_group}:
   {3'd0,2'(R),9'(batch_x+12'(P)),output_group};
  wire [7:0] pw_scalar=scalar_a[R][P];
  wire [7:0] rgb_a=pw_scalar,rgb_b=scalar_b[R][P];
  wire [63:0] ordinary;
  if(R%2==0)assign ordinary=stride_rotated[R/2][P];
  else if(P%3==0)assign ordinary=dw_rotation[R][P/3];
  else assign ordinary=(P%3==1 ? dw_rem0:dw_rem_lt2) ? dw_rotation[R][(P/3+1)%3]:dw_rotation[R][P/3];
  wire [63:0] dw_word;
  if(P==0)assign dw_word=dw_age==0 ? history[0][R*64+:64]:dw_age==3 ? left_column[(R+1)*64+:64]:dw_age==6 ? left_column[(R+2)*64+:64]:ordinary;
  else assign dw_word=ordinary;
  wire [63:0] stride_word;
  if(DP<9)assign stride_word=stride_rotated[DR][DP];
  else assign stride_word=64'd0;
  // Choose once per byte class before replication. RGB/PW share the same
  // low-three phase bits; all legal PW term masks have low bits 111.
  // RGB stem has exactly three input channels. subphase=3 is unreachable;
  // avoid selecting a fourth generic byte that this operator never uses.
  wire [7:0] stem_scalar=subphase==0 ? dw_word[7:0]:subphase==1 ? dw_word[15:8]:dw_word[23:16];
  wire [7:0] repeated_byte=pw_mode || rgb ? pw_scalar:stem_scalar;
  wire [7:0] repeat_a=use_repeat ? repeated_byte:8'd0;
  wire [7:0] repeat_b=use_repeat ? (rgb ? rgb_b:repeated_byte):8'd0;
  wire [7:0] repeat_c=use_repeat && !rgb ? repeated_byte:8'd0;
  // Even rows share the complete routing network. P0 alone has the DW
  // left-edge/history correction; stride must bypass that correction.
  wire [63:0] selected_word;
  if(R%2==0)begin:SHARED_WORD
   wire [63:0] common_word=P==0 && !use_stride ? dw_word:ordinary;
   assign selected_word={64{use_dw || use_stride}} & common_word;
  end else begin:ODD_WORD
   assign selected_word=({64{use_dw}} & dw_word) | ({64{use_stride}} & stride_word);
  end
  for(genvar l=0;l<8;l=l+1)begin:DATA_LANE
   if(l<3)assign pool_data[o*64+l*8+:8]=selected_word[l*8+:8] | repeat_a;
   else if(l<6)assign pool_data[o*64+l*8+:8]=selected_word[l*8+:8] | repeat_b;
   else assign pool_data[o*64+l*8+:8]=selected_word[l*8+:8] | repeat_c;
  end
  wire [63:0] normal_weight,pair_weight;
  if(P==0)assign normal_weight=weight_data;else assign normal_weight=ordinary_dw && stem ? weights[3*P-1]:weights[P-1];
  if(DP==0)assign pair_weight=weight_data;else if(DP<9)assign pair_weight=weights[DP-1];else assign pair_weight=0;
  assign pool_weights[o*64+:64]=paired && !rgb ? pair_weight : rgb ? {16'd0,normal_weight[23:0],normal_weight[23:0]}:normal_weight;
  always @(posedge clk)begin
   if(reset || cancel)begin owners[o]<=0;end
   else if(phase_advance)begin
    if(column_advance)begin if(first)owners[o]<=launch;else if(last)owners[o]<=0;end
   end
  end
 end
 wire [1:0] log_in=cmd_in_groups==4 ? 2'd2:cmd_in_groups==2 ? 2'd1:2'd0;
 wire [1:0] log_out=cmd_out_groups==4 ? 2'd2:cmd_out_groups==2 ? 2'd1:2'd0;
 wire good=cmd_pixels>=1 && cmd_pixels<=320 && cmd_rows>=1 && cmd_rows<=(cmd_pair && !cmd_rgb ? 2:4) &&
  (cmd_in_groups==1 || cmd_in_groups==2 || cmd_in_groups==4) && (cmd_out_groups==1 || cmd_out_groups==2 || cmd_out_groups==4) && cmd_mask!=0;
 always @(posedge clk)begin
  if(reset || cancel)begin
   running<=0;done<=0;error<=0;paired<=0;rgb<=0;pw_mode<=0;stem<=0;warming<=0;pixels<=0;batch<=0;batches<=0;base<=0;
   rows<=0;group_q<=0;log_cin<=0;subphase<=0;phase<=0;phase_mask<=0;term_mask<=0;terms<=0;mask<=0;next_column<=0;next_weight<=0;acked<=0;
   left_column<=0;cycles<=0;wait_input_cycles<=0;wait_pool_cycles<=0;
  end else begin
   if(done_valid && done_ready)done<=0;
   if(cmd_valid && cmd_ready)begin
    running<=good;done<=!good;error<=!good;paired<=cmd_pair;rgb<=cmd_rgb;pw_mode<=cmd_mode_pw;stem<=cmd_stem;
    warming<=!cmd_pair && !cmd_mode_pw;pixels<=cmd_pixels;rows<=cmd_rows;group_q<=cmd_group;log_cin<=log_in;
    batch<=0;base<=0;batches<=9'(({1'b0,cmd_pixels}+10'd7)>>3);subphase<=0;phase<=0;
    phase_mask<=cmd_pair ? (cmd_rgb ? 7'd7:7'd8) : cmd_mode_pw ? 7'((8<<(log_in+{1'b0,log_out}))-1):7'd8;
    terms<=6'(8<<log_in);term_mask<=5'((8<<log_in)-1);mask<=cmd_mask;next_column<=0;next_weight<=0;acked<=0;
    cycles<=0;wait_input_cycles<=0;wait_pool_cycles<=0;
   end
   if(active)begin
    cycles<=cycles+1'b1;
    if(!tokens)wait_input_cycles<=wait_input_cycles+1'b1;else if(!warming && !phase_advance)wait_pool_cycles<=wait_pool_cycles+1'b1;
    if(input_fire)begin
     if(in_index!=next_column || (ordinary_dw && in_warmup!=warming))error<=1;
     next_column<=next_column+13'd1;
     if(warming)warming<=0;
    end
    if(phase_advance)begin
     if(weight_index!=next_weight)error<=1;
     next_weight<=next_weight+13'd1;acked<=0;
     subphase<=ordinary_dw && stem && subphase!=2 ? subphase+2'd1:2'd0;
     if(column_advance)begin
      if(ordinary_dw && phase==0)left_column<=history[0][383:0];
      if(phase==phase_mask)begin phase<=0;batch<=batch+9'd1;base<=base+(rgb ? 9'd8:9'd9);end else phase<=phase+7'd1;
     end
    end else acked<=acked|(pool_valid&pool_ready);
    if(!warming && (paired ? base>=units:batch>=batches) && owners==0 && acked==0)begin running<=0;done<=1;end
   end
  end
 end
endmodule
