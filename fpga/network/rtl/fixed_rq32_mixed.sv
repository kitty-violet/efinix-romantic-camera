// One set of32 RQ DSPs for MAC, residual alignment and residual post-sum.
// Separate bounded credits/queues break the feedback dependency cycle.
// Priority post > align > MAC cannot create unbounded feedback traffic:
// each MAC vector creates at most one alignment and one post-sum vector.
module fixed_rq32_mixed #(parameter integer REGISTER_SCALE_CLASS=0,SHARED_FEEDBACK=0)(
 input wire clk,reset,cancel,
 input wire [2:0] in_valid,output wire [2:0] in_ready,
 input wire [767:0] mac_sums,input wire [255:0] align_value,input wire [287:0] post_value,
 input wire [26:0] in_x,input wire [287:0] feedback_value,
 output wire feedback_post_selected,
 input wire [255:0] bias,multiplier,input wire [7:0] relu_enable,mask,
 input wire [17:0] align_multiplier,post_multiplier,
 output wire [2:0] out_valid,input wire [2:0] out_ready,
 output wire [767:0] out_data,output wire [26:0] out_x,output wire [2:0] out_bad,
 output wire busy,output reg error
);
 reg [3:0] reserved[0:2],stored[0:2];reg [3:0] token_valid,bias_bad;
 reg [1:0] kinds[0:3];reg [8:0] xpipe[0:3];reg [7:0] masks[0:3];
 wire active=!reset && !cancel && !error;wire [2:0] eligible;
 for(genvar s=0;s<3;s=s+1)assign eligible[s]=active && in_valid[s] && reserved[s]<8;
 // Only unusually large INT32 values with small scales need a low-bit
 // correction. The existing DSPs perform an uncommitted internal first pass.
 // The producer retains its vector until the eventual ready handshake.
 reg preparing;reg [1:0] correction_wait;
 // Job selection is a separate clock boundary. Payload remains with its
 // producer until in_ready, so no second wide data buffer is required.
 reg selected;reg [1:0] selected_kind;reg [31:0] selected_split;
 wire [31:0] split_lane;wire issue;
 wire first_fire,dispatch_selected,first_pending;
 wire [607:0] shared_a;wire [575:0] shared_c;wire [639:0] shared_hi;

 // Producer hint is NOT a reserved token. Accept only live valid + credit.
 // Other producers retain priority; a vanished hint is cleared/reselected.
 reg selected_hint;
 wire [2:0] reserve_eligible=eligible & (issue ? ~(3'b001<<selected_kind):3'b111);
 always @(posedge clk)begin
  if(reset || cancel)begin selected<=0;selected_kind<=0;selected_split<=0;selected_hint<=0;end
  else if(active)begin
   if(!selected || issue || selected_hint)begin
    if(|reserve_eligible)begin
     selected<=1;selected_kind<=reserve_eligible[2] ? 2'd2:reserve_eligible[1] ? 2'd1:2'd0;
     selected_split<=32'd0;selected_hint<=0;
    end else if(issue || selected_hint)begin selected<=1;selected_hint<=1;end
    else begin selected<=0;selected_hint<=0;end
   end
  end
 end
 // Production holds this parameter view before the first MAC result and
 // until the last MAC acceptance. Classify each shared channel one clock
 // ahead; feedback jobs use their independent signed18 scale.
 wire [7:0] scale_fits7_now,scale_nonzero_now,scale_invalid_now;
 reg [7:0] scale_fits7_q,scale_nonzero_q,scale_invalid_q;
 for(genvar c=0;c<8;c=c+1)begin:SCALE_CLASS
  assign scale_fits7_now[c]=multiplier[c*32+7+:11]=={11{multiplier[c*32+6]}};
  assign scale_nonzero_now[c]=|multiplier[c*32+:18];
  assign scale_invalid_now[c]=multiplier[c*32+18+:14]!={14{multiplier[c*32+17]}};
 end
 always @(posedge clk)begin
  scale_fits7_q<=scale_fits7_now;
  scale_nonzero_q<=scale_nonzero_now;
  scale_invalid_q<=scale_invalid_now;
 end
 wire [7:0] scale_fits7=REGISTER_SCALE_CLASS!=0 ? scale_fits7_q:scale_fits7_now;
 wire [7:0] scale_nonzero=REGISTER_SCALE_CLASS!=0 ? scale_nonzero_q:scale_nonzero_now;
 wire [7:0] scale_invalid=REGISTER_SCALE_CLASS!=0 ? scale_invalid_q:scale_invalid_now;
 // Classify the stable header bias once per channel. The signed four-bit
 // partial carry remains fully general (-8..7); no numeric range is removed.
 // Thresholds share -bias and ~bias low nibbles. Classification is ready
 // under the same pre-MAC header contract as REGISTER_SCALE_CLASS.
 // BIAS_RANGE_CLASS_BEGIN
 function automatic [12:0] classify_bias_range(input signed [11:0] h);
  begin
   classify_bias_range[3:0]=4'd0-h[3:0];
   classify_bias_range[7:4]=~h[3:0];
   classify_bias_range[8]=(h>=-12'sd7 && h<=12'sd8);
   classify_bias_range[9]=(h>=-12'sd8 && h<=12'sd7);
   classify_bias_range[10]=(h>=-12'sd8 && h<=12'sd8);
   classify_bias_range[11]=(h>=12'sd9 && h<=12'sd23);
   classify_bias_range[12]=(h>=-12'sd23 && h<=-12'sd9);
  end
 endfunction
 // BIAS_RANGE_CLASS_END
 wire [103:0] bias_range_now;reg [103:0] bias_range_q;
 for(genvar c=0;c<8;c=c+1)begin:BIAS_RANGE
  assign bias_range_now[c*13+:13]=classify_bias_range($signed(bias[c*32+20+:12]));
 end
 always @(posedge clk)bias_range_q<=bias_range_now;
 wire [103:0] bias_range=REGISTER_SCALE_CLASS!=0 ? bias_range_q:bias_range_now;
 // All four spatial rows share a channel's scale. If any needs the exact
 // split path, use that path for the channel: small rows remain exact too.
 // This removes four separate 18-bit scale selectors per channel without
 // adding correction cycles (the same original split_lane starts the work).
 wire [7:0] split_channel;
 for(genvar c=0;c<8;c=c+1)assign split_channel[c]=selected_split[c] || selected_split[8+c] || selected_split[16+c] || selected_split[24+c];
 wire [1:0] chosen=selected_kind;
 // Operand steering comes from the already registered job. Late
 // cancel/valid/credit only qualify issue, never the arithmetic data.
 assign feedback_post_selected=chosen==2'd2;
 wire selected_eligible=selected && eligible[chosen];
 wire need_correction=chosen==0 && |selected_split;
 // Data steering uses only reserved state, independent of late cancel and
 // loader validity. Actual issue/ready still enforce every original guard.
 wire correction_operand=need_correction && !preparing;
 wire start_correction=selected_eligible && correction_operand;
 // Reuse the existing32 DSPs for the exceptional low-bit product. A/B
 // capture at edge0, W at edge1, O at edge2. At edge3 the high product
 // captures the exact low correction directly from O into C.
 // No normal output token is issued for that internal first pass.
 assign issue=selected_eligible;
 always @(posedge clk)begin
  if(reset || cancel)begin preparing<=0;correction_wait<=0;end
  else if(active)begin
   if(start_correction)begin preparing<=1;correction_wait<=2;end
   else if(preparing && correction_wait!=0)correction_wait<=correction_wait-1'b1;
   if(issue)preparing<=0;
  end
 end
 // Feedback jobs never need the MAC's INT32 range/correction decision.
 // Keep these grants independent so a 13-bit bias carry and 32-lane range
 // reduction cannot become a skip/RAM/writer READY path. The MAC grant and
 // shared token issue retain their exact original cycle.
 (* syn_keep="true" *) wire post_grant=selected_eligible && chosen==2;
 (* syn_keep="true" *) wire align_grant=selected_eligible && chosen==1;
 assign in_ready={post_grant,align_grant,(issue && chosen==0)};
 // synthesis translate_off
 always @(posedge clk)if(active && in_ready!==(issue ? (3'b001<<chosen):3'd0))
  $fatal(1,"RQ_READY_FACTORIZATION_MISMATCH");
 // synthesis translate_on
 wire [31:0] rq_valid,rq_range,overflow;wire [255:0] rq_data,qualified;
 wire [7:0] selected_mask=chosen==0 ? mask:8'hff;
 wire [17:0] feedback_multiplier=chosen==2 ? post_multiplier:align_multiplier;
 // A channel's multiplier selection is identical across all four rows.
 // Preserve that shared combinational boundary instead of mapping four muxes.
 wire [143:0] channel_scale;
 for(genvar c=0;c<8;c=c+1)begin:SHARED_SCALE
  (* syn_keep = "true" *) wire [17:0] selected_scale=correction_operand ?
   {{11{multiplier[c*32+6]}},multiplier[c*32+:7]}:chosen==0 ?
   (split_channel[c] ? {multiplier[c*32+:12],6'd0}:multiplier[c*32+:18]):feedback_multiplier;
  assign channel_scale[c*18+:18]=selected_scale;
 end
 shared_rq_phase96 shared_phase(.clk(clk),.rst_n(!reset && !cancel),.selected(selected),
  .candidate(selected_eligible),.accepted(issue),.first_fire(first_fire),
  .dispatch_selected(dispatch_selected),.first_pending(first_pending));
 shared_rq_dsp96 shared_dsp(.clk(clk),.first_fire(first_fire),.pending(first_pending),.rst_n({16{!reset && !cancel}}),
  .operand_a(shared_a),.scale(channel_scale),.rounded_c(shared_c),.product_hi(shared_hi));
 // synthesis translate_off
 always @(posedge clk)if(!reset && !cancel && selected_eligible && need_correction)
  $fatal(1,"RQ96_TARGET_REQUIRES_NONZERO_SIGNED7_REJECTION");
 // synthesis translate_on
 for(genvar lane=0;lane<32;lane=lane+1)begin:LANE
  localparam C=lane%8;
  // The low20 bias was accumulated with all channel phases. Its signed
  // carry rejoins the shared signed upper bias here, with an overflow bit.
  wire signed [23:0] low=mac_sums[lane*24+:24];
  wire signed [11:0] high_bias=bias[C*32+20+:12];
  wire signed [12:0] high=$signed({high_bias[11],high_bias})+$signed({{9{low[23]}},low[23:20]});
  wire signed [32:0] sum={high,low[19:0]};
  wire signed [7:0] a=align_value[lane*8+:8];wire signed [8:0] p=post_value[lane*9+:9];
  // Decode MAC range once before choosing the feedback operand. Feedback
  // already fits signed9; a 32-bit mux followed by repeated range decoding
  // is redundant. Keep every overflow, signed18-scale and rounding rule.
  wire normal=chosen==0;
  wire [12:0] range_class=bias_range[C*13+:13];
  // BIAS_RANGE_LANE_BEGIN
  wire signed [3:0] carry=low[23:20];
  wire fits19=(low[19]==low[18]) &&
   ((!low[18] && range_class[8] && carry==range_class[3:0]) ||
    ( low[18] && range_class[9] && carry==range_class[7:4]));
  wire fits25=range_class[10] ||
   (range_class[11] && carry<=$signed(range_class[7:4])) ||
   (range_class[12] && carry>=$signed(range_class[3:0]));
  // BIAS_RANGE_LANE_END
  // synthesis translate_off
  always @(posedge clk)if(active && in_valid[0])begin
   if({fits19,fits25}!=={sum[31:19]=={13{sum[18]}},sum[31:25]=={7{sum[24]}}})
    $fatal(1,"RQ_BIAS_RANGE_NOT_PREPARED");
  end
  // synthesis translate_on
  wire fits7=scale_fits7[C];
  assign split_lane[lane]=!fits19 && fits25 && fits7;
  assign overflow[lane]=chosen==0 && mask[C] && (sum>33'sd2147483647 || sum< -33'sd2147483648);
  wire split=normal && split_channel[C];
  wire [18:0] mac_operand=split_channel[C] ? sum[24:6]:sum[18:0];
  // The fused producer selects its operand only for the accepted job.
  // Arbitration depends on valid/credits, never on this payload.
  wire signed [8:0] feedback=SHARED_FEEDBACK!=0 ? feedback_value[lane*9+:9]:
   chosen==1 ? {a[7],a}:p;
  wire [18:0] feedback_operand={{10{feedback[8]}},feedback};
  wire [18:0] operand_a=correction_operand ? {13'd0,low[5:0]}:normal ? mac_operand:feedback_operand;
  wire [17:0] operand_b=channel_scale[C*18+:18];
  wire saturated=normal && !fits19 && !split_lane[lane] && scale_nonzero[C];
  wire negative=normal ? sum[31]^multiplier[C*32+17]:feedback[8]^feedback_multiplier[17];
  wire range_bad=normal && scale_invalid[C];
  assign shared_a[lane*19+:19]=operand_a;
  assign shared_c[lane*18+:18]=negative ? 18'd65535:18'd65536;
  fixed_rq_predecoded_pipeline #(.EXTERNAL_PRODUCT(1)) rq(.product_high(shared_hi[lane*20+:20]),.operand_a(operand_a),.operand_b(operand_b),
   .correction_enable(split && preparing),.saturated(saturated),.negative(negative),.range_bad(range_bad),
   .clk(clk),.rst_n(!reset && !cancel),.in_valid(issue),
   .relu_enable(normal && relu_enable[C]),.out_valid(rq_valid[lane]),.result(rq_data[lane*8+:8]),.multiplier_range_error(rq_range[lane]));
  assign qualified[lane*8+:8]=masks[3][C] && !rq_range[lane] ? rq_data[lane*8+:8]:8'd0;
 end
 wire [265:0] payload={bias_bad[3] || (|(rq_range & {4{masks[3]}})),xpipe[3],qualified};
 // All three queues share their input. Use the native inverted Q and
 // complement the one shared input bus, preserving initial output too.
 (* syn_keep="true" *) wire [265:0] inverted_payload=~payload;
 wire [2:0] queue_bad,nonempty;
 for(genvar s=0;s<3;s=s+1)begin:QUEUE
  wire push=active && token_valid[3] && kinds[3]==2'(s);
  wire [265:0] front;wire [2:0] address=3'd0-stored[s][2:0];
  for(genvar b=0;b<266;b=b+1)begin:BIT
   wire q;EFX_SRL8 #(.INIT(8'hff)) storage(.D(inverted_payload[b]),.A(address),.CLK(clk),.CE(push),.Q(q),.Q7());assign front[b]=q;
  end
  assign out_valid[s]=active && stored[s]!=0;assign {out_bad[s],out_x[s*9+:9],out_data[s*256+:256]}=front;
  wire pop=out_valid[s] && out_ready[s];wire accept=issue && chosen==2'(s);
  assign queue_bad[s]=reserved[s]>8 || stored[s]>8 || (push && stored[s]==8 && !pop) || (pop && reserved[s]==0);
  assign nonempty[s]=reserved[s]!=0 || stored[s]!=0;
  always @(posedge clk)begin
   if(reset || cancel)begin reserved[s]<=0;stored[s]<=0;end
   else if(!error)begin
    case({accept,pop})2'b10:reserved[s]<=reserved[s]+4'd1;2'b01:reserved[s]<=reserved[s]-4'd1;default:reserved[s]<=reserved[s];endcase
    case({push,pop})2'b10:stored[s]<=stored[s]+4'd1;2'b01:stored[s]<=stored[s]-4'd1;default:stored[s]<=stored[s];endcase
   end
  end
 end
 assign busy=(selected && (!selected_hint || (|eligible))) || preparing || (|nonempty)||(|token_valid);
 always @(posedge clk)begin
  if(reset || cancel)begin
   token_valid<=0;bias_bad<=0;error<=0;
   for(integer i=0;i<4;i=i+1)begin kinds[i]<=0;xpipe[i]<=0;masks[i]<=0;end
  end else if(!error)begin
   token_valid<={token_valid[2:0],issue};bias_bad<={bias_bad[2:0],|overflow};
   kinds[0]<=chosen;xpipe[0]<=chosen==0 ? in_x[8:0]:chosen==1 ? in_x[17:9]:in_x[26:18];masks[0]<=selected_mask;
   for(integer i=1;i<4;i=i+1)begin kinds[i]<=kinds[i-1];xpipe[i]<=xpipe[i-1];masks[i]<=masks[i-1];end
   if(rq_valid!={32{token_valid[3]}} || (|queue_bad) || (selected && !selected_hint && !in_valid[chosen]))error<=1;
  end
 end
 // synthesis translate_off
 always @(posedge clk)if(REGISTER_SCALE_CLASS!=0 && active &&
   ((issue && chosen==0) || start_correction))begin
  if({scale_fits7_q,scale_nonzero_q,scale_invalid_q}!==
     {scale_fits7_now,scale_nonzero_now,scale_invalid_now})
   $fatal(1,"RQ_SCALE_CLASS_NOT_PREPARED");
 end
 // synthesis translate_on
endmodule


// Exact INT32 * signed18 Q17, unchanged four-edge public latency.
// Move operand/correction registers into the existing DSP, then use its
// output register to retain the original pipeline depth. No extra DSP.
module fixed_rq_predecoded_pipeline #(parameter integer EXTERNAL_PRODUCT=0)(
 input wire [19:0] product_high,
 input wire clk,rst_n,in_valid,
 input wire signed [18:0] operand_a,input wire signed [17:0] operand_b,
 input wire saturated,negative,range_bad,relu_enable,
 input wire correction_enable,
 output reg out_valid,
 output reg signed [7:0] result,
 output reg multiplier_range_error
);
 reg sat0,neg0,relu0,err0,v0;
 reg sat1,neg1,relu1,err1,v1;
 reg sat2,neg2,relu2,err2,v2;
 wire signed [47:0] product2;
 // Internal low-bit pass carries its own rounding bias. On the high-bit
 // pass reuse the full corrected C field; no fabric correction adder and
 // no post-product increment are needed. The full-product sign supplied
 // by the decoder also handles a negative high part with positive low bits.
 wire signed [17:0] rounded_c=correction_enable ? product2[17:0]:
  (negative ? 18'sd65535:18'sd65536);
 generate if(EXTERNAL_PRODUCT==0)begin:LOCAL_DSP
 EFX_DSP48 #(.A_REG(1),.B_REG(1),.C_REG(1),.P_REG(0),.W_REG(1),.O_REG(1),
  .RST_SYNC(1),.SIGNED(1),.P_EXT("ALIGN_RIGHT"),.C_EXT("ALIGN_RIGHT"),
  .M_SEL("P"),.N_SEL("C"),.W_SEL("X"),.CASCOUT_SEL("W")) multiply_add (
  .A(operand_a),.B(operand_b),.C(rounded_c),.CASCIN(48'd0),
  .OP(2'b00),.SHIFT_ENA(1'b0),.CLK(clk),.CE(1'b1),.RST(!rst_n),
  .O(product2),.CASCOUT(),.OVFL());
 end else begin assign product2={{11{product_high[19]}},product_high,17'd0};end endgenerate
 // Signed19 * signed18 plus signed14 is exactly representable in signed37.
 // Upper DSP bits only sign-extend that result; verify the bound in four-state gates.
 // synthesis translate_off
 always @(posedge clk)if(rst_n && v2 && product2[47:36]!=={12{product2[36]}})
  $fatal(1,"RQ_PRODUCT_EXCEEDS_SIGNED37");
 // synthesis translate_on
 function automatic signed [7:0] rounded;
  input signed [36:0] p;
  input relu;
 
  begin
   if(p[36] && relu)rounded=0;
   else if(!p[36])rounded=(|p[35:24]) ? 8'sd127:$signed(p[24:17]);
   else rounded=!(&p[35:24]) ? -8'sd128:$signed(p[24:17]);
  end
 endfunction
 always @(posedge clk)begin
  sat0<=saturated;neg0<=negative;
  relu0<=relu_enable;err0<=range_bad;
  sat1<=sat0;neg1<=neg0;relu1<=relu0;err1<=err0;
  sat2<=sat1;neg2<=neg1;relu2<=relu1;err2<=err1;
 end
 always @(posedge clk)begin
  if(!rst_n)begin
   v0<=0;v1<=0;v2<=0;out_valid<=0;result<=0;multiplier_range_error<=0;
  end else begin
   v0<=in_valid;v1<=v0;v2<=v1;out_valid<=v2;
   if(v2)begin
    result<=sat2 ? ((neg2 && relu2) ? 8'sd0:neg2 ? -8'sd128:8'sd127):rounded(product2[36:0],relu2);
    multiplier_range_error<=err2;
   end
  end
 end
endmodule

// Two halves use16 existing DSP48s. The producer is acknowledged only on
// the second half; all metadata/queue boundaries keep the old accept latency.
// Current target already rejects nonzero signed7 MAC scales, so no split pass.
module shared_rq_phase96(input clk,rst_n,selected,candidate,accepted,output first_fire,dispatch_selected,first_pending);
assign first_fire=0;assign dispatch_selected=selected;assign first_pending=0;
endmodule
module shared_rq_dsp96(input clk,first_fire,pending,input [15:0]rst_n,input [607:0]operand_a,input[143:0]scale,input[575:0]rounded_c,output[639:0]product_hi);
function automatic[7:0]clip(input[19:0]h);
 if(h[19])clip=(&h[18:7])?h[7:0]:8'h80;else clip=(|h[18:7])?8'h7f:h[7:0];
endfunction
for(genvar l=0;l<32;l=l+1)begin:LANE
wire[47:0]p;wire[7:0]c=clip(p[36:17]);
EFX_DSP48 #(.A_REG(1),.B_REG(1),.C_REG(1),.P_REG(0),.W_REG(1),.O_REG(1),.RST_SYNC(1),.SIGNED(1),.P_EXT("ALIGN_RIGHT"),.C_EXT("ALIGN_RIGHT"),.M_SEL("P"),.N_SEL("C"),.W_SEL("X"),.CASCOUT_SEL("W")) mult(
.A(operand_a[l*19+:19]),.B(scale[(l%8)*18+:18]),.C(rounded_c[l*18+:18]),.CASCIN(48'd0),.OP(2'b00),.SHIFT_ENA(1'b0),.CLK(clk),.CE(1'b1),.RST(!rst_n[l%16]),.O(p),.CASCOUT(),.OVFL());
assign product_hi[l*20+:20]=l<16?{{12{c[7]}},c}:p[36:17];
end
endmodule
