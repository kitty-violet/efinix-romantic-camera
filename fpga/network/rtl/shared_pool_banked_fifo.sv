// Fixed dispatch schedule: <=one LAST per bank per phase; reject collisions.
// Credits include accepted LAST through DSP tail, RQ and result FIFO.
// MAC tails have priority; fused N32 phases leave bounded feedback service gaps.
// MAC results and residual feedback have independent queues/credits: no cyclic HOL dependency.
// Production private tags use <=15 bits; bit15 is rejected then reused for operation kind inside FIFO.
// Geometry tags use 13 bits today. Public frame/image identities remain outside this queue.
// Isolated per-bank fair mixed MAC/residual RQ service; operation kind follows each token.
// Isolated production-pool candidate:32 MAC fronts,4 shared RQs,4 actual
// RAM-backed result queues. No private per-front result payload copies or
//32-to4 output collector.16 credits/bank include every in-flight RQ token.
// Private16-bit job tags; public frame/coordinates belong to caller descriptors.
// Registered credit eligibility breaks RAM-ready -> pop -> MAC-ready -> RAM
// request/collision loops. A full-bank simultaneous pop is reusable next cycle.
// Cancellation is GLOBAL and synchronously flushes every dependent pipeline.
// Residual bypass uses the SAME four RQ pipelines and result FIFOs.
// Mode changes only on an idle configuration handshake; signed9 inputs preserve
// the exact aligned-byte sum (-256..254). Normal MAC/front protocol is unchanged.
// Production dispatch pairs o/o+16 have identical term timing whenever
// the second owner participates. Each MAC and result credit remains private.
module shared_pool_banked_fifo #(parameter integer PAIRED_SCHEDULE=0,EXTERNAL_HEADER=0)(
 input wire clk,reset,cancel,epoch_clear,
 output wire header_first_valid,output reg [1:0] header_first_group,input wire header_first_ready,
 output wire header_use_valid,output wire [1:0] header_use_group,input wire header_use_ready,
 input wire cfg_valid,output wire cfg_ready,output wire cfg_loaded,
 input wire [7:0] cfg_result_mask,
 input wire cfg_mixed,
 input wire cfg_bypass,input wire [71:0] cfg_bypass_multiplier,
 input wire [3:0] bypass_valid,output wire [3:0] bypass_ready,
 input wire [287:0] bypass_value,input wire [63:0] bypass_tag,input wire [31:0] bypass_mask,
 input wire [1023:0] cfg_bias,cfg_multiplier,input wire [31:0] cfg_relu,input wire [2:0] cfg_groups,
 input wire [31:0] in_valid,output wire [31:0] in_ready,input wire [31:0] in_first,in_last,
 input wire [191:0] in_terms,input wire [2047:0] in_activation,in_weights,
 input wire [63:0] in_group,input wire [511:0] in_tag,input wire [255:0] in_mask,
 output wire [3:0] out_valid,output wire [3:0] out_bypass,input wire [3:0] out_ready,output wire [255:0] out_data,
 output wire [63:0] out_tag,output wire [31:0] out_mask,out_bias_overflow,out_range_error,
 output wire [3:0] feedback_valid,input wire [3:0] feedback_ready,output wire [255:0] feedback_data,
 output wire [63:0] feedback_tag,
 output wire [31:0] finish_request,finish_grant,output wire busy,output wire error,
 output wire [19:0] reserved_count
);
 wire run;reg [1:0] shared_group;
 reg mixed_mode;reg bypass_mode;reg [71:0] bypass_multipliers;
 reg [7:0] result_mask;reg [3:0] result_fault;
 reg loaded,cfg_error;reg [1023:0] biases;wire [1023:0] multipliers;
 reg [575:0] compact_scales;reg [31:0] scale_errors;wire configuring;
 for(genvar m=0;m<32;m=m+1)begin:SCALE
  wire sign_bit=compact_scales[m*18+17];
  assign multipliers[m*32+:32]={{13{sign_bit}},(scale_errors[m] ? !sign_bit:sign_bit),compact_scales[m*18+:18]};
  always @(posedge clk)begin
   if(reset)begin compact_scales[m*18+:18]<=0;scale_errors[m]<=0;end
   else if(configuring && cfg_result_mask!=0 && !(cfg_mixed && cfg_bypass) && (cfg_groups==1 || cfg_groups==2 || cfg_groups==4))begin
    if(EXTERNAL_HEADER==0)compact_scales[m*18+:18]<=cfg_multiplier[m*32+:18];
    if(EXTERNAL_HEADER==0)scale_errors[m]<=cfg_multiplier[m*32+18+:14]!={14{cfg_multiplier[m*32+17]}};
   end
  end
 end
 reg [31:0] relus;reg [2:0] groups;
 wire flush=reset || cancel;
 wire [31:0] front_busy,front_error,front_ready,dot_valid,accept_enable;
 wire [31:0] first_candidates=in_valid & in_first;
 assign header_first_valid=EXTERNAL_HEADER!=0 && run && (|first_candidates);
 always @*begin
  header_first_group=0;
  for(integer p=0;p<32;p=p+1)header_first_group=header_first_group|(in_group[p*2+:2]&{2{first_candidates[p]}});
 end
 wire header_step=EXTERNAL_HEADER==0 || !(|first_candidates) || header_first_ready;
 assign header_use_valid=EXTERNAL_HEADER!=0 && (|finish_request);
 assign header_use_group=shared_group;
 reg header_fault;
 // Latch the notification; feeding it combinationally into global flush would
 // loop through front.reset -> finish_request -> this condition. Refuse the
 // RQ issue immediately below, then cancel the row on the following edge.
 always @(posedge clk)begin
  if(flush)header_fault<=0;
  else if(EXTERNAL_HEADER!=0 && header_use_valid && !header_use_ready)header_fault<=1;
 end
 wire [5375:0] dot_previous;
 wire [8191:0] dot_data;wire [63:0] request_group,dot_group;
 wire [511:0] request_tag;wire [255:0] request_mask,dot_mask;
 wire [3:0] cluster_busy,cluster_error,queue_empty,queue_error,credit_error,bank_last_credit;
 wire all_last_credit=&bank_last_credit;
 reg group_collision;
 reg [255:0] shared_bias,shared_multiplier;reg [7:0] shared_relu;
 always @*begin
  shared_group=0;
  for(integer o=0;o<32;o=o+1)shared_group=shared_group | (request_group[o*2+:2] & {2{finish_request[o]}});
  group_collision=0;
  for(integer o=0;o<32;o=o+1)if(finish_request[o] && request_group[o*2+:2]!=shared_group)group_collision=1;
  shared_bias=0;shared_multiplier=0;shared_relu=0;
  if(EXTERNAL_HEADER!=0)begin
   shared_bias=cfg_bias[255:0];shared_multiplier=cfg_multiplier[255:0];shared_relu=cfg_relu[7:0];
  end else for(integer g=0;g<4;g=g+1)begin
   shared_bias=shared_bias | (biases[g*256+:256] & {256{shared_group==2'(g)}});
   shared_multiplier=shared_multiplier | (multipliers[g*256+:256] & {256{shared_group==2'(g)}});
   shared_relu=shared_relu | (relus[g*8+:8] & {8{shared_group==2'(g)}});
  end
 end
 assign busy=!reset && ((|front_busy)||(|cluster_busy)||!( &queue_empty)||(|reserved_count));
 assign error=header_fault || (|result_fault) || cfg_error || (|front_error) || (|cluster_error) || (|queue_error) || (|credit_error);
 assign cfg_ready=!flush && !busy && !error;
 assign cfg_loaded=!flush && loaded && !error;
 assign configuring=cfg_valid && cfg_ready;
 assign run=cfg_loaded && !configuring;
 assign in_ready=front_ready & accept_enable;
 wire [31:0] control_fire,control_last,control_opened;
 for(genvar o=0;o<32;o=o+1)begin:FRONT
  localparam integer SHARE=(PAIRED_SCHEDULE!=0 && o>=16);
  localparam integer LEADER=o%16;
  shared_mac_stream_front #(.SHARED_SCHEDULE(SHARE),.EXTERNAL_PREVIOUS(1)) front(.clk(clk),.reset(flush),.epoch_clear(configuring || epoch_clear),
   .schedule_fire(SHARE ? control_fire[LEADER]:1'b0),.schedule_first(SHARE ? in_first[LEADER]:1'b0),
   .schedule_last(SHARE ? control_last[LEADER]:1'b0),.schedule_terms(SHARE ? in_terms[LEADER*6+:6]:6'd0),
   .control_fire(control_fire[o]),.control_last(control_last[o]),.control_opened(control_opened[o]),.in_valid(in_valid[o] && accept_enable[o]),.in_ready(front_ready[o]),
   .in_first(in_first[o]),.in_last(in_last[o]),.in_terms(in_terms[o*6+:6]),
   .in_activation(in_activation[o*64+:64]),.in_weights(in_weights[o*64+:64]),
   .in_group(in_group[o*2+:2]),.in_tag(in_tag[o*16+:16]),.in_mask(in_mask[o*8+:8]),
   .request(finish_request[o]),.grant(finish_grant[o]),.request_group(request_group[o*2+:2]),
   .request_tag(request_tag[o*16+:16]),.request_mask(request_mask[o*8+:8]),
   .dot_previous(dot_previous[o*168+:168]),.dot_valid(dot_valid[o]),.dot_data(dot_data[o*256+:256]),.dot_group(dot_group[o*2+:2]),.dot_mask(dot_mask[o*8+:8]),
   .busy(front_busy[o]),.error(front_error[o]));
 end
 for(genvar b=0;b<4;b=b+1)begin:BANK
  wire selected_valid,mac_selected_valid;reg [2:0] selected;
  reg [4:0] mac_reserved;reg [3:0] feedback_reserved;wire [4:0] reserved=mac_reserved+{1'b0,feedback_reserved};
  reg bad_credit;
  wire has_mac=|finish_request[b*8+:8];
  wire choose_bypass=!has_mac && (bypass_mode || mixed_mode);
  wire [7:0] last_fire=in_valid[b*8+:8] & in_ready[b*8+:8] & in_last[b*8+:8];
  wire reserve_mac=|last_fire;
  wire collision=|(last_fire & (last_fire-8'd1)) || |(finish_request[b*8+:8] & (finish_request[b*8+:8]-8'd1));
  assign bank_last_credit[b]=!(|(in_valid[b*8+:8] & in_last[b*8+:8])) || mac_reserved<16;
  assign accept_enable[b*8+:8]={8{run && header_step && (!bypass_mode || mixed_mode)}} &
   (~in_last[b*8+:8] | {8{all_last_credit}});
  assign mac_selected_valid=run && has_mac && (EXTERNAL_HEADER==0 || header_use_ready);
  assign finish_grant[b*8+:8]=finish_request[b*8+:8] & {8{run}};
  always @*begin
   selected=0;
   for(integer o=0;o<8;o=o+1)selected=selected | (3'(o) & {3{finish_request[b*8+o]}});
  end
  assign bypass_ready[b]=run && (bypass_mode || mixed_mode) && choose_bypass && feedback_reserved<8;
  assign selected_valid=choose_bypass ? (bypass_valid[b] && bypass_ready[b]) : mac_selected_valid;
  assign reserved_count[b*5+:5]=reserved;assign credit_error[b]=bad_credit;
  wire pop=out_valid[b] && out_ready[b];
  wire feedback_pop=feedback_valid[b] && feedback_ready[b];
  reg [25:0] request_meta;
  always @*begin
   request_meta=0;
   for(integer o=0;o<8;o=o+1)request_meta=request_meta | ({26{run && finish_request[b*8+o]}} &
    {request_tag[(b*8+o)*16+:16],request_group[(b*8+o)*2+:2],request_mask[(b*8+o)*8+:8]});
   if(choose_bypass)request_meta={bypass_tag[b*16+:16],2'd0,bypass_mask[b*8+:8]};
  end
  wire returned,killed;wire [63:0] data;wire [7:0] mask,bias_flag,range_flag;
  wire [2:0] returned_owner;wire [1:0] returned_slot;wire [7:0] returned_generation;
  shared_rq_oc8_cluster #(.NUM_FRONTS(8),.OWNER_WIDTH(3),.DIRECT_DOT_OWNER(1),.PRESELECTED_HEADER(1),.BANK_PREFIX_HISTORY(1),.REJECT_BIAS_OVERFLOW(1)) rq(.clk(clk),.reset(flush),.epoch_clear(configuring || epoch_clear),
   .bypass_mode(choose_bypass),.bypass_value(bypass_value[b*72+:72]),.bypass_multiplier(bypass_multipliers[b*18+:18]),
   .issue_valid(selected_valid),.issue_owner(choose_bypass ? 3'd0 : selected),.issue_slot(2'd0),.issue_generation(8'd0),
   .issue_group(request_meta[9:8]),.issue_layer_id(8'd0),.issue_image_tag(16'd0),.issue_expected_mask(request_meta[7:0]),
   .owner_cancel_pending(8'd0),.dot_valid(dot_valid[b*8+:8]),.dot_data(dot_data[b*2048+:2048]),.dot_previous(dot_previous[b*1344+:1344]),
   .dot_slot(16'd0),.dot_generation(64'd0),.dot_group(dot_group[b*16+:16]),.dot_layer_id(64'd0),
   .dot_image_tag(128'd0),.dot_expected_mask(dot_mask[b*64+:64]),
   .header_bias({768'd0,shared_bias}),.header_multiplier({768'd0,shared_multiplier}),.header_relu({24'd0,shared_relu}),.header_groups(groups),
   .header_layer_id(8'd0),.header_image_tag(16'd0),.return_valid(returned),.return_killed(killed),
   .return_owner(returned_owner),.return_slot(returned_slot),.return_generation(returned_generation),
   .return_data(data),.return_mask(mask),.return_bias_overflow(bias_flag),.return_range_error(range_flag),
   .busy(cluster_busy[b]),.protocol_error(cluster_error[b]));
  reg [63:0] tag_pipe;reg [3:0] tag_valid;
  reg [3:0] kind_pipe;wire [79:0] fifo_data,feedback_word;wire queue_valid,feedback_queue_valid;
  wire mac_empty,feedback_empty,mac_queue_error,feedback_queue_error;
  wire result_bad=returned && (killed || mask!=result_mask || (|bias_flag) || (|range_flag) || tag_pipe[63]);
  assign out_valid[b]=queue_valid && !error && !flush;
  banked_result_fifo #(.WIDTH(80)) queue(.clk(clk),.reset(flush),.push(returned && !kind_pipe[3] && !error && !result_bad),
   .in_data({kind_pipe[3],tag_pipe[62:48],data}),
   .out_valid(queue_valid),.out_ready(out_ready[b] && !error),.out_data(fifo_data),.empty(mac_empty),.error(mac_queue_error));
  shallow_result_fifo #(.WIDTH(80)) feedback_queue(.clk(clk),.reset(flush),
   .push(returned && kind_pipe[3] && !error && !result_bad),.in_data({1'b0,tag_pipe[62:48],data}),
   .out_valid(feedback_queue_valid),.out_ready(feedback_ready[b] && !error),.out_data(feedback_word),
   .empty(feedback_empty),.error(feedback_queue_error));
  assign feedback_valid[b]=feedback_queue_valid && !error && !flush;
  assign feedback_data[b*64+:64]=feedback_word[63:0];assign feedback_tag[b*16+:16]={1'b0,feedback_word[78:64]};
  assign queue_empty[b]=mac_empty && feedback_empty;assign queue_error[b]=mac_queue_error || feedback_queue_error;
  assign out_bypass[b]=1'b0;assign out_tag[b*16+:16]={1'b0,fifo_data[78:64]};assign out_data[b*64+:64]=fifo_data[63:0];
  assign out_range_error[b*8+:8]=0;assign out_bias_overflow[b*8+:8]=0;assign out_mask[b*8+:8]=result_mask;
  always @(posedge clk)begin
   if(flush)result_fault[b]<=0;else if(result_bad)result_fault[b]<=1;
  end
  always @(posedge clk)begin
   if(flush)begin kind_pipe<=0;mac_reserved<=0;feedback_reserved<=0;bad_credit<=0;tag_pipe<=0;tag_valid<=0;end
   else begin
    kind_pipe<={kind_pipe[2:0],choose_bypass};
    tag_valid<={tag_valid[2:0],selected_valid};tag_pipe<={tag_pipe[47:0],request_meta[25:10]};
    if(group_collision || collision || returned!=tag_valid[3] || (returned && killed) || (pop && mac_reserved==0) || (feedback_pop && feedback_reserved==0))bad_credit<=1;
    case({reserve_mac,pop})2'b10:mac_reserved<=mac_reserved+5'd1;2'b01:mac_reserved<=mac_reserved-5'd1;default:mac_reserved<=mac_reserved;endcase
    case({(choose_bypass && selected_valid),feedback_pop})2'b10:feedback_reserved<=feedback_reserved+4'd1;2'b01:feedback_reserved<=feedback_reserved-4'd1;default:feedback_reserved<=feedback_reserved;endcase
    if(mac_reserved>16 || feedback_reserved>8)bad_credit<=1;
   end
  end
 end
 always @(posedge clk)begin
  if(reset)begin result_mask<=0;mixed_mode<=0;bypass_mode<=0;bypass_multipliers<=0;loaded<=0;cfg_error<=0;biases<=0;relus<=0;groups<=0;end
  else if(cancel)cfg_error<=0;
  else if(configuring)begin
   if(cfg_result_mask!=0 && !(cfg_mixed && cfg_bypass) && (cfg_groups==1 || cfg_groups==2 || cfg_groups==4))begin
    result_mask<=cfg_result_mask;mixed_mode<=cfg_mixed;loaded<=1;bypass_mode<=cfg_bypass;bypass_multipliers<=cfg_bypass_multiplier;if(EXTERNAL_HEADER==0)begin biases<=cfg_bias;relus<=cfg_relu;end groups<=cfg_groups;
   end else begin loaded<=0;cfg_error<=1;end
  end else if(epoch_clear && busy)cfg_error<=1;
 end
endmodule
