// One production8-channel group: real coordinates/MAC/shared residual RQ/
// acknowledged writeback with DMA sharing the real six B ports.
module fixed_video_compute #(parameter integer SKIP_BASE_WIDTH=10,SKIP_OFFSET_PREAPPLIED=0)(
 input wire clk,reset,cancel,
 input wire weight_begin,output wire weight_begin_ready,input wire [2:0] weight_parts,
 input wire weight_valid,output wire weight_ready,input wire [5:0] weight_index,
 input wire [63:0] weight_data,input wire weight_last,
 input wire cmd_valid,output wire cmd_ready,input wire cmd_pw,cmd_stem,cmd_stride2,cmd_nn,cmd_stride4,cmd_nn4,
 input wire [8:0] cmd_pixels,input wire [2:0] cmd_rows,cmd_groups,input wire [1:0] cmd_group,
 input wire [10:0] cmd_output_x,cmd_source_width,cmd_source_origin,input wire cmd_storage_phase,
 input wire [17:0] cmd_row_banks,input wire [59:0] cmd_row_bases,
 input wire [2:0] cmd_out_groups,input wire [1:0] cmd_out_group,
 input wire [11:0] cmd_out_banks,input wire [39:0] cmd_out_bases,input wire [3:0] cmd_out_phase,
 input wire cmd_fused,cmd_main_scaled,cmd_skip_offset_two,
 input wire [11:0] cmd_skip_banks,input wire [4*SKIP_BASE_WIDTH-1:0] cmd_skip_bases,
 input wire [255:0] bias,multiplier,input wire [7:0] relu_enable,mask,
 input wire [17:0] align_multiplier,post_multiplier,
 output wire done_valid,input wire done_ready,output wire busy,output reg error,
 output wire mac_inputs_done,
 output wire [5:0] read_valid,input wire [5:0] read_ready,output wire [59:0] read_address,
 input wire [5:0] read_response,output wire [5:0] read_response_ready,input wire [767:0] read_data,
 input wire [5:0] dma_req_valid,output wire [5:0] dma_req_ready,input wire [59:0] dma_req_address,
 input wire [11:0] dma_req_half,input wire [767:0] dma_req_data,
 output wire [5:0] dma_rsp_valid,input wire [5:0] dma_rsp_ready,output wire [767:0] dma_rsp_data,output wire [5:0] dma_rsp_write,
 output wire [5:0] aux_write_collision,
 output wire [5:0] aux_valid,input wire [5:0] aux_ready,output wire [59:0] aux_address,
 output wire [11:0] aux_write_half,output wire [767:0] aux_data,
 input wire [5:0] aux_response,output wire [5:0] aux_response_ready,input wire [767:0] aux_response_data,input wire [5:0] aux_was_write,
 output wire flush_arena,fabric_busy,fabric_error,output wire [12:0] writes,acks,
 output wire [12:0] accepted_vectors,output wire [8:0] results
);
 localparam IDLE=0,RUN=1,ABORT=2,DONE=3;
 // Production adapter owns the descriptor until DONE. Header prefetch writes
 // the other group slot, so the active parameter view remains stable.
 // Do not instantiate this specialized variant behind a transient command bus.
 reg [1:0] state;
 // Runtime exact integer fast path; generic arithmetic remains available.
 // Q17(1 * signed8) is zero and Q17(131071 * signed8) is identity.
 // Descriptor shape/address rejection is retained before bypassing skip reads.
 reg identity_descriptor_ok;
 always @* begin
  identity_descriptor_ok=cmd_pw && cmd_groups==4 && cmd_out_groups==4 && mask==255 &&
    cmd_rows>=1 && cmd_rows<=3 && cmd_pixels>=1 && cmd_pixels<=320;
  for(integer r=0;r<4;r=r+1)if(r<cmd_rows)begin
   if(cmd_skip_banks[r*3+:3]>=6)identity_descriptor_ok=0;
   if((int'(cmd_skip_bases[r*SKIP_BASE_WIDTH+:SKIP_BASE_WIDTH])+
      ((int'(cmd_pixels)-1+(SKIP_OFFSET_PREAPPLIED ? 0:(cmd_skip_offset_two ? 2:0)))>>1)*4+
      int'(cmd_out_group))>=1024)identity_descriptor_ok=0;
   for(integer t=0;t<r;t=t+1)
    if(cmd_skip_banks[t*3+:3]==cmd_skip_banks[r*3+:3])identity_descriptor_ok=0;
  end
 end
 wire identity_residual=cmd_fused && !cmd_main_scaled && align_multiplier==18'd1 &&
   post_multiplier==18'd131071 && identity_descriptor_ok;
 // One decision at existing command acceptance; no extra execution state.
 // The wide descriptor predicate never drives the live RQ/writeback select.
 reg fused;
 always @(posedge clk)begin
  if(reset)fused<=0;
  else if(start)fused<=cmd_fused && !identity_residual;
 end
 wire main_scaled=cmd_main_scaled,skip_offset_two=cmd_skip_offset_two;
 wire [8:0] pixels=cmd_pixels;wire [2:0] rows=cmd_rows,groups=cmd_out_groups;
 wire [1:0] group_id=cmd_out_group;
 wire [11:0] out_banks=cmd_out_banks,skip_banks=cmd_skip_banks;
 wire [39:0] out_bases=cmd_out_bases;wire [4*SKIP_BASE_WIDTH-1:0] skip_bases=cmd_skip_bases;
 wire [3:0] out_phase=cmd_out_phase;
 wire [255:0] bias_q=bias,multiplier_q=multiplier;
 wire [7:0] relu_q=relu_enable,mask_q=mask;
 wire [17:0] align_q=align_multiplier,post_q=post_multiplier;
 // Each producer withdraws its requests on its registered fault. Flush the
 // shared RAM owner on that same boundary; waiting for ABORT adds a cycle
 // in which a held request disappears without cancellation. all_error only
 // combines registered faults; no ready/response combinational path is added.
 wire all_error;
 wire local_flush=cancel || state==ABORT || all_error;assign flush_arena=local_flush;
 wire raw_ready,raw_wbr,raw_wr,raw_valid,raw_take,raw_done,raw_error;wire [767:0] sums;wire [8:0] raw_x;
 assign mac_inputs_done=raw_done;
 wire rq_busy,rq_error,skip_busy,skip_error,writer_busy,writer_error;
 assign all_error=raw_error || rq_error || skip_error || writer_error || fabric_error;
 wire clean=!rq_busy && !skip_busy && !writer_busy;
 assign cmd_ready=!reset && !cancel && state==IDLE && raw_ready;
 wire start=cmd_valid && cmd_ready;
 // Historical weights remain owned until arithmetic drain completes.
 // The ring may stage the next group without destroying the active history.
 assign weight_begin_ready=state!=ABORT && raw_wbr;assign weight_ready=state!=ABORT && raw_wr;
 fixed_production_mac #(.TAIL_SHIFT_048(0)) mac(.bias(bias_q),.clk(clk),.reset(reset),.cancel(local_flush),
  .weight_begin(weight_begin && state!=ABORT),.weight_begin_ready(raw_wbr),.weight_parts(weight_parts),
  .weight_valid(weight_valid && state!=ABORT),.weight_ready(raw_wr),.weight_index(weight_index),.weight_data(weight_data),.weight_last(weight_last),
  .cmd_valid(start),.cmd_ready(raw_ready),.out_valid(raw_valid),.out_ready(raw_take),.out_sums(sums),.out_x(raw_x),
  .done_valid(raw_done),.done_ready(state==RUN && clean),.busy(),.error(raw_error),.*);
 wire [2:0] iv,ir,ov,ordy,ob;wire [767:0] od;wire [26:0] ox,ix;
 wire av,ar,pv,pr,mr,alr;wire [255:0] align_value;wire [287:0] post_value;wire [8:0] ax,px;
 wire [287:0] feedback_value;wire feedback_post_selected;
 wire wrq;wire backend_reset=reset || start;
 assign iv={pv,av,raw_valid};assign raw_take=ir[0];assign {pr,ar}=ir[2:1];assign ix={px,ax,raw_x};
 assign ordy=fused ? {wrq,alr,mr}:{2'b00,wrq};
 // The descriptor/header owner prepares scale before the first MAC result.
 fixed_rq32_mixed #(.REGISTER_SCALE_CLASS(1),.SHARED_FEEDBACK(1)) rq(.feedback_post_selected(feedback_post_selected),.feedback_value(feedback_value),.clk(clk),.reset(backend_reset),.cancel(local_flush),.in_valid(iv),.in_ready(ir),.mac_sums(sums),.align_value(align_value),.post_value(post_value),
  .in_x(ix),.bias(bias_q),.multiplier(multiplier_q),.relu_enable(relu_q),.mask(mask_q),.align_multiplier(align_q),.post_multiplier(post_q),
  .out_valid(ov),.out_ready(ordy),.out_data(od),.out_x(ox),.out_bad(ob),.busy(rq_busy),.error(rq_error));
 // synthesis translate_off
 always @(posedge clk)if(!backend_reset && !local_flush)begin
  if(ir[1] || ir[2])begin
   if(feedback_post_selected!==ir[2])$fatal(1,"FEEDBACK_KIND_GUARD_MISMATCH");
   for(integer q=0;q<32;q=q+1)begin
    if(ir[2] && feedback_value[q*9+:9]!==post_value[q*9+:9])$fatal(1,"FEEDBACK_POST_OPERAND_MISMATCH");
    if(ir[1] && feedback_value[q*9+:9]!=={align_value[q*8+7],align_value[q*8+:8]})$fatal(1,"FEEDBACK_ALIGN_OPERAND_MISMATCH");
   end
  end
 end
 // synthesis translate_on
 wire [5:0] wv,wr,ws,wsr,wsw,sv,sr,ss,ssr,ssw;wire [59:0] wa,sa;wire [11:0] wh;wire [767:0] wd,sd;
 fixed_skip_vectors4 #(.BASE_WIDTH(SKIP_BASE_WIDTH),.OFFSET_PREAPPLIED(SKIP_OFFSET_PREAPPLIED),.REGISTERED_FEEDBACK_SELECT(1)) skip(.feedback_post_selected(feedback_post_selected),.feedback_value(feedback_value),.clk(clk),.reset(backend_reset),.cancel(local_flush),.enable(fused && state==RUN),.main_scaled(main_scaled),
  .pixels(pixels),.rows(rows),.group_id(group_id),.skip_banks(skip_banks),.skip_bases(skip_bases),.offset_two(skip_offset_two),
  .mac_valid(ov[0] && fused),.mac_ready(mr),.mac_data(od[255:0]),.mac_x(ox[8:0]),.mac_bad(ob[0]),
  .align_valid(av),.align_ready(ar),.align_value(align_value),.align_x(ax),
  .aligned_valid(ov[1]),.aligned_ready(alr),.aligned_data(od[511:256]),.aligned_x(ox[17:9]),.aligned_bad(ob[1]),
  .post_valid(pv),.post_ready(pr),.post_value(post_value),.post_x(px),
  .request_valid(sv),.request_ready(sr),.request_address(sa),.response_valid(ss),.response_ready(ssr),.response_data(sd),.response_was_write(ssw),.busy(skip_busy),.error(skip_error));
 // One shared result selection before the six destination-bank selectors.
 // Retain the 256-bit common mux boundary through technology mapping.
 (* syn_keep = "true" *) wire [255:0] writer_values=fused ? od[767:512]:od[255:0];
 fixed_column_store4 #(.REGISTER_ADDRESSES(1)) writer(.prepare(start),.clk(clk),.reset(backend_reset),.cancel(local_flush),.pixels(pixels),.rows(rows),.groups(groups),.group_id(group_id),
  .row_banks(out_banks),.row_bases(out_bases),.row_phase(out_phase),.in_valid(fused ? ov[2]:ov[0]),.in_ready(wrq),
  .in_data(writer_values),.in_x(fused ? ox[26:18]:ox[8:0]),.in_bad(fused ? ob[2]:ob[0]),
  .request_valid(wv),.request_ready(wr),.request_address(wa),.request_half(wh),.request_data(wd),
  .response_valid(ws),.response_ready(wsr),.response_was_write(wsw),.busy(writer_busy),.error(writer_error),.writes(writes),.acks(acks));
 wire [17:0] reqready,rspvalid,rspwrite;wire [2303:0] rspdata;
 assign {dma_req_ready,sr,wr}=reqready;assign {dma_rsp_valid,ss,ws}=rspvalid;
 assign {dma_rsp_write,ssw,wsw}=rspwrite;assign dma_rsp_data=rspdata[1536+:768];assign sd=rspdata[768+:768];
 row_pair_aux_router #(.READ_ONLY_SKIP(1)) router(.clk(clk),.reset(reset),.cancel(local_flush),.request_valid({dma_req_valid,sv,wv}),.request_ready(reqready),
  .protected_read_fire(read_valid & read_ready),.protected_read_address(read_address),.aux_write_collision(aux_write_collision),
  .request_address({dma_req_address,sa,wa}),.request_half({dma_req_half,12'd0,wh}),.request_data({dma_req_data,768'd0,wd}),
  .response_valid(rspvalid),.response_ready({dma_rsp_ready,ssr,wsr}),.response_data(rspdata),.response_was_write(rspwrite),
  .aux_valid(aux_valid),.aux_ready(aux_ready),.aux_address(aux_address),.aux_write_half(aux_write_half),.aux_data(aux_data),
  .aux_response(aux_response),.aux_response_ready(aux_response_ready),.aux_response_data(aux_response_data),.aux_was_write(aux_was_write),
  .busy(fabric_busy),.error(fabric_error));
 wire [12:0] expected_writes=rows==1 ? {4'd0,pixels}:rows==2 ? {3'd0,pixels,1'b0}:rows==3 ? {4'd0,pixels}+{3'd0,pixels,1'b0}:{2'd0,pixels,2'd0};
 assign busy=state!=IDLE;
 wire finishing=state==RUN && raw_done && clean && !all_error && writes==expected_writes && acks==expected_writes;
 assign done_valid=!reset && !cancel && (state==DONE || finishing);
 always @(posedge clk)begin
  if(reset)begin
   state<=IDLE;error<=0;
  end else begin
   if(state==RUN && (cancel || all_error))begin state<=ABORT;error<=1;end
   else case(state)
    IDLE:if(start)begin
     state<=RUN;error<=0;
     if((cmd_fused && (!cmd_pw || cmd_groups!=4 || cmd_out_groups!=4 || mask!=255)) ||
        (!(cmd_out_groups==1 || cmd_out_groups==2 || cmd_out_groups==4)) || {1'b0,cmd_out_group}>=cmd_out_groups ||
        (!cmd_pw && !cmd_stem && cmd_out_group!=cmd_group))begin state<=ABORT;error<=1;end
    end
    RUN:if(raw_done && clean)begin
     if(writes!=expected_writes || acks!=expected_writes)begin state<=ABORT;error<=1;end else state<=done_ready ? IDLE:DONE;
    end
    ABORT:if(!cancel)state<=DONE;
    DONE:if(done_ready)state<=IDLE;
    default:state<=IDLE;
   endcase
  end
 end
 // synthesis translate_off
 wire [123+4*SKIP_BASE_WIDTH:0] descriptor_view={cmd_fused,main_scaled,skip_offset_two,pixels,rows,groups,group_id,
  out_banks,skip_banks,out_bases,skip_bases,out_phase,align_q,post_q};
 wire [527:0] parameter_view={bias_q,multiplier_q,relu_q,mask_q};
 reg [123+4*SKIP_BASE_WIDTH:0] held_descriptor;
 reg [527:0] held_parameters;
 always @(posedge clk)begin
  if(start)begin held_descriptor<=descriptor_view;held_parameters<=parameter_view;end
  if(!reset && !cancel && state==RUN)begin
   if(descriptor_view!==held_descriptor)$fatal(1,"FIXED_OWNER_CONFIG_CHANGED_WHILE_RUNNING");
   // No MAC request can remain after raw_done. Align/post RQ do not use
   // bias/multiplier/relu/mask, so the next group's header may now be read.
   if(!raw_done && parameter_view!==held_parameters)$fatal(1,"FIXED_HEADER_CHANGED_BEFORE_LAST_MAC");
   if(raw_done && raw_valid)$fatal(1,"FIXED_MAC_AFTER_DONE");
  end
 end
 // synthesis translate_on
endmodule
