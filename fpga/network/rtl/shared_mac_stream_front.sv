// Scheduled private front: LAST reserves bank credit; every valid tail is consumed.
// Prefix snapshots stay in private mailboxes; subtraction is shared downstream.
// Production global-cancel front. Output buffering lives ONCE per RQ bank.
// Keeps the exact selected MAC/prefix implementation and fixed3-edge dot tail.
// Private production option: owners o and o+16 share accepted term progress.
// The secondary keeps its own transaction identity, pending state and MAC tail.
// Missing/extra secondary tokens or mismatched starts fail closed.
module shared_mac_stream_front #(parameter integer SHARED_SCHEDULE=0,EXTERNAL_PREVIOUS=0)(
 input wire clk,reset,epoch_clear,
 input wire schedule_fire,schedule_first,schedule_last,input wire [5:0] schedule_terms,
 output wire control_fire,control_last,control_opened,
 input wire in_valid,output wire in_ready,input wire in_first,in_last,input wire [5:0] in_terms,
 input wire [63:0] in_activation,in_weights,input wire [1:0] in_group,
 input wire [15:0] in_tag,input wire [7:0] in_mask,
 output wire request,input wire grant,
 output wire [1:0] request_group,output wire [15:0] request_tag,output wire [7:0] request_mask,
 output wire [167:0] dot_previous,
 output wire dot_valid,output wire [255:0] dot_data,output reg [1:0] dot_group,output reg [7:0] dot_mask,
 output wire busy,output reg error
);
 reg opened,pending;reg [15:0] result_tag;reg [5:0] count,terms;
 reg [1:0] group_q;reg [15:0] tag_q;reg [7:0] mask_q;
 wire [3:0] valids,errors;wire [255:0] accum;
 wire last_expected=opened && ((SHARED_SCHEDULE!=0) ? schedule_last : count==terms-1'b1);
 assign control_last=last_expected;assign control_opened=opened;
 // Raw results already live in the MAC output registers. Publish that mailbox
 // independently of the next dot; do not serialize simultaneous LAST terms.
 assign request=!reset && !error && pending && (&valids);
 assign in_ready=!reset && !error;
 wire fire=in_valid && in_ready;
 assign control_fire=fire;
 wire schedule_bad=(SHARED_SCHEDULE!=0) &&
  ((fire && (!schedule_fire || in_first!=schedule_first || (in_first && in_terms!=schedule_terms))) ||
   (opened && schedule_fire && !fire));
 wire legal=in_terms==8 || in_terms==9 || in_terms==16 || in_terms==27 || in_terms==32;
 wire bad=schedule_bad || (fire && ((!opened && (!in_first || in_last || !legal || in_mask==0)) ||
  (opened && (in_first || in_last!=last_expected))));
 assign request_group=dot_group;assign request_tag=result_tag;assign request_mask=dot_mask;
 assign dot_valid=request;
 assign busy=opened || pending || (|valids) || error;
 for(genvar p=0;p<4;p=p+1)begin:MAC
  int8_dual_mac_lane_prefix_stream #(.EXTERNAL_PREVIOUS(EXTERNAL_PREVIOUS)) mac(.clk(clk),.rst_n(!reset),.clear(1'b0),.epoch_clear(epoch_clear),.consume(grant),.previous_pair(dot_previous[p*42+:42]),
   .in_valid(fire && !bad),.first_term(!opened),.last_term(last_expected),
   .activation0(in_activation[p*16+:8]),.weight0(in_weights[p*16+:8]),
   .activation1(in_activation[p*16+8+:8]),.weight1(in_weights[p*16+8+:8]),
   .dot_valid(valids[p]),.dot0(accum[p*64+:32]),.dot1(accum[p*64+32+:32]),.protocol_error(errors[p]));
 end
 // Same eight fixed sign extensions without the nested DOT generate alias.
 assign dot_data={{{11{accum[244]}},accum[224+:21]},
  {{11{accum[212]}},accum[192+:21]},
  {{11{accum[180]}},accum[160+:21]},
  {{11{accum[148]}},accum[128+:21]},
  {{11{accum[116]}},accum[96+:21]},
  {{11{accum[84]}},accum[64+:21]},
  {{11{accum[52]}},accum[32+:21]},
  {{11{accum[20]}},accum[0+:21]}};
 always @(posedge clk)begin
  if(reset || (EXTERNAL_PREVIOUS!=0 && epoch_clear))begin opened<=0;pending<=0;result_tag<=0;count<=0;terms<=0;group_q<=0;tag_q<=0;mask_q<=0;dot_group<=0;dot_mask<=0;error<=0;end
  else begin
   if(bad || (|errors) || (grant && !request))error<=1;
   if(grant)pending<=0;
   if(fire && !bad)begin
    if(!opened)begin opened<=1;if(SHARED_SCHEDULE==0)begin count<=1;terms<=in_terms;end group_q<=in_group;tag_q<=in_tag;mask_q<=in_mask;end
    else if(last_expected)begin
     opened<=0;pending<=1;result_tag<=tag_q;dot_group<=group_q;dot_mask<=mask_q;
     if(pending && !grant)error<=1;
    end else if(SHARED_SCHEDULE==0)count<=count+1'b1;
   end
   if(|valids)begin
    if(valids!=4'hf || !pending || !grant)error<=1;
   end
  end
 end
endmodule
