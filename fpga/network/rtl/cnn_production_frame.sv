// Actual hardware program reader + finite issuer + CNN + shared read merge.
// One external CNN AXI master. No local memory responder or traffic generator.
// Host coefficient load/readback remains a real interface; board boot/MMIO is
// separate. This module alone is not the camera/display/PHY complete baseline.
module cnn_production_frame #(parameter integer RESIDENT_PROGRAMS=0)(
 input wire  clk,
 input wire  reset,
 input wire  cancel,
 input wire discard,output reg discarded,
 input wire [2:0] active_style_slot,
 input wire  recover,
 output wire  recover_ready,
 output wire  quiesced,
 output wire  halted,
 output wire  needs_common_reset,
 output wire [7:0] fault_code,
 output wire  busy,
 output wire  error,
 input wire [15:0] job_image_tag,
 input wire  begin_valid,
 output wire  begin_ready,
 input wire [31:0] begin_version,
 input wire [31:0] begin_image_bytes,
 input wire [15:0] begin_tag,
 input wire  load_valid,
 output wire  load_ready,
 input wire [11:0] load_index,
 input wire [63:0] load_data,
 input wire  load_last,
 input wire  verify_valid,
 output wire  verify_ready,
 input wire [15:0] verify_tag,
 input wire  verify_ok,
 input wire  diag_valid,
 output wire  diag_accept,
 input wire [2:0] diag_bank,
 input wire [9:0] diag_address,
 output wire  diag_response,
 input wire  diag_ready,
 output wire [63:0] diag_data,
 output wire [3:0] diag_error,
 output wire [2:0] diag_return_bank,
 output wire [9:0] diag_return_address,
 output wire [15:0] diag_return_tag,
 output wire  coeff_ready,
 output wire  loading,
 output wire  load_complete,
 output wire  store_error,
 output wire [3:0] error_code,
 output wire [15:0] image_tag,
 output wire [11:0] loaded_words,
 output wire [31:0] araddr,
 output wire [31:0] awaddr,
 output wire [7:0] arlen,
 output wire [7:0] awlen,
 output wire  arvalid,
 output wire  awvalid,
 output wire  rready,
 output wire  wlast,
 output wire  wvalid,
 output wire  bready,
 input wire  arready,
 input wire  awready,
 input wire  wready,
 input wire  rvalid,
 input wire  rlast,
 input wire  bvalid,
 input wire [127:0] rdata,
 input wire [1:0] rresp,
 input wire [1:0] bresp,
 output wire [127:0] wdata,
 output wire [15:0] wstrb,
 output wire [31:0] cycles,
 output wire [31:0] reads,
 output wire [31:0] writes,
 output wire [31:0] acks,
 output wire [31:0] dcycles,
 output wire [31:0] dwords,
 output wire [31:0] dbeats,
 output wire  arb_busy,
 output wire  arena_router_busy,
 input wire start_valid,output wire start_ready,
 input wire [31:0] compute_crc32,dma_crc32,
 output wire frame_valid,output wire [3:0] program_fault_code,
 output wire [13:0] compute_completed,dma_completed,
 output wire [28:0] nodes_completed,output wire [19:0] rgb_pixels,
 output wire [31:0] production_cycles
);
 wire cfg_storage_phase;wire [3:0] cfg_out_phase;
 wire  cfg_valid;
 wire  cfg_ready;
 wire  done_valid;
 wire  done_ready;
 wire  cfg_pw_fuse;
 wire  cfg_main_scaled;
 wire cfg_stride2,cfg_stride4,cfg_nn4;
 wire  cfg_stem;
 wire  cfg_mode_pw;
 wire  cfg_nn_view;
 wire [11:0] cfg_skip_banks;
 wire [39:0] cfg_skip_bases;
 wire [10:0] cfg_skip_delta;
 wire [17:0] align_multiplier;
 wire [17:0] post_multiplier;
 wire [8:0] cfg_pixels;
 wire [2:0] cfg_rows;
 wire [2:0] cfg_in_groups;
 wire [2:0] cfg_groups;
 wire [1:0] cfg_group;
 wire [7:0] cfg_mask;
 wire [10:0] cfg_output_x;
 wire [10:0] cfg_source_width;
 wire [10:0] cfg_source_origin;
 wire [17:0] cfg_row_banks;
 wire [59:0] cfg_row_bases;
 wire [11:0] out_banks;
 wire [39:0] out_bases;
 wire [14:0] cfg_weight_offset;
 wire  pcv;
 wire  pcr;
 wire  hv;
 wire  hready;
 wire [14:0] hbias;
 wire [14:0] hmult;
 wire [14:0] hrelu;
 wire [7:0] hlayer;
 wire [2:0] hgroups;
 wire  hsvalid;
 wire  hssuccess;
 wire  hsready;
 wire  hbusy;
 wire  herror;
 wire  dvalid;
 wire  dready;
 wire  dstore;
 wire  drgb;
 wire [31:0] daddress;
 wire [8:0] dcolumns;
 wire [2:0] dgroups;
 wire [2:0] dbank;
 wire [9:0] dbase;
 wire  ddone;
 wire  ddready;
 wire  derror;
 wire  dbusy;
 wire  dp_busy;
 wire  dp_error;
 wire [31:0] dp_araddr;
 wire [7:0] dp_arlen;
 wire  dp_arvalid;
 wire  dp_rready;
 wire  dp_arready;
 wire  dp_rvalid;
 wire  dp_rlast;
 wire [127:0] dp_rdata;
 wire [1:0] dp_rresp;
 wire cv,cr,dv,dr,nv,fetch_busy,fetch_verified,fetch_error,issue_busy,issue_error,fetch_start_ready;
 wire [511:0] cdata,ndata;wire [127:0] ddata;wire [3:0] fetch_fault;
 wire [31:0] fetch_araddr;wire [7:0] fetch_arlen;wire fetch_arvalid,fetch_arready,fetch_rready;
 wire [255:0] merged_rdata;wire [3:0] merged_rresp;wire [1:0] merged_rvalid,merged_rlast,merged_arready;
 wire merge_busy,compute_begin,compute_end,dma_end;wire [14:0] debug_program_index;
 wire dp_recover_ready,dp_quiesced,dp_needs_common_reset;
 reg datapath_fault_q;
 reg discard_pending;wire issue_discarded,issue_frame_valid,issue_discard_boundary;
 wire discard_request=discard_pending || discard;
 // Keep program records alive until current compute and DMA commands finish.
 wire discard_fetch=discard_request && (issue_discard_boundary || !issue_busy) && !arb_busy && !arena_router_busy;
 always @(posedge clk)begin
  if(reset)begin discard_pending<=0;discarded<=0;end
  else begin
   discarded<=0;
   if(discarded)discard_pending<=0;
   else if(discard)discard_pending<=1;
   if(discard_pending && quiesced && !error && !datapath_fault_q && !cancel)begin
    discarded<=1;discard_pending<=0;
   end
  end
 end
 assign frame_valid=issue_frame_valid && !discard_request && !discarded;
 // Program faults use a common transport reset. Do not advertise local core
 // recovery while the program reader may still own a DDR transaction.
 assign needs_common_reset=dp_needs_common_reset || issue_error || fetch_error || datapath_fault_q;
 assign recover_ready=dp_recover_ready && !needs_common_reset && !fetch_busy && !merge_busy;
 assign quiesced=!issue_busy && !fetch_busy && !merge_busy && !dp_busy && !dbusy && !hbusy && !arb_busy && !arena_router_busy;
 // The datapath rejects an illegal offered command combinationally. Register
 // its notification before gating the producer, otherwise VALID -> reject ->
 // producer CANCEL -> VALID forms a loop. The datapath itself blocks the bad
 // request immediately, before this notification edge.
 always @(posedge clk)begin
  if(reset)datapath_fault_q<=0;else if(dp_error)datapath_fault_q<=1;
 end
 wire allowed=!discard_request && !discarded && !issue_busy && !dp_busy && !dbusy && !hbusy && !loading && coeff_ready && !halted;
 assign start_ready=fetch_start_ready && allowed;
 wire start_fire=start_valid && start_ready;
 assign busy=issue_busy || fetch_busy || dp_busy || dbusy || hbusy || merge_busy;
 assign error=issue_error || fetch_error || dp_error;
 frame_program_fetch #(.RESIDENT_PROGRAMS(RESIDENT_PROGRAMS)) fetch(.active_style_slot(active_style_slot),.clk(clk),.reset(reset),.cancel(cancel || issue_error || datapath_fault_q),.graceful_abort(discard_fetch),
  .start_valid(start_valid && allowed),.start_ready(fetch_start_ready),
  .compute_records(14'd6546),.dma_records(14'd8277),.compute_crc32(compute_crc32),.dma_crc32(dma_crc32),
  .cvalid(cv),.cready(cr),.cdata(cdata),.next_valid(nv),.next_data(ndata),
  .dvalid(dv),.dready(dr),.ddata(ddata),
  .araddr(fetch_araddr),.arlen(fetch_arlen),.arvalid(fetch_arvalid),.arready(fetch_arready),
  .rdata(merged_rdata[255:128]),.rresp(merged_rresp[3:2]),.rlast(merged_rlast[1]),.rvalid(merged_rvalid[1]),.rready(fetch_rready),
  .busy(fetch_busy),.quiesced(),.verified(fetch_verified),.error(fetch_error),.fault_code(fetch_fault),.compute_fetched(),.dma_fetched());
 frame_program_issue #(.AUTO_DW_GROUPS(1)) issue(
  .clk(clk),
  .reset(reset),
  .cancel(cancel),.discard(discard_request),.discarded(issue_discarded),
  .discard_boundary(issue_discard_boundary),.discard_release(!fetch_busy && !merge_busy && !arb_busy && !arena_router_busy),
  .start(start_fire),
  .compute_records(14'd6546),
  .dma_records(14'd8277),
  .cvalid(cv),
  .cready(cr),
  .cdata(cdata),
  .next_valid(nv),
  .next_data(ndata),
  .dvalid(dv),
  .dready(dr),
  .ddata(ddata),
  .fetch_verified(fetch_verified),
  .fetch_error(fetch_error),
  .cfg_valid(cfg_valid),
  .cfg_ready(cfg_ready),
  .done_valid(done_valid),
  .done_ready(done_ready),
  .datapath_busy(dp_busy),
  .datapath_error(datapath_fault_q),
  .cfg_pw_fuse(cfg_pw_fuse),
  .cfg_main_scaled(cfg_main_scaled),
  .cfg_stride2(cfg_stride2),.cfg_stride4(cfg_stride4),.cfg_nn4(cfg_nn4),
  .cfg_stem(cfg_stem),
  .cfg_mode_pw(cfg_mode_pw),
  .cfg_nn_view(cfg_nn_view),
  .cfg_skip_banks(cfg_skip_banks),
  .cfg_skip_bases(cfg_skip_bases),
  .cfg_skip_delta(cfg_skip_delta),
  .align_multiplier(align_multiplier),
  .post_multiplier(post_multiplier),
  .cfg_pixels(cfg_pixels),
  .cfg_rows(cfg_rows),
  .cfg_in_groups(cfg_in_groups),
  .cfg_groups(cfg_groups),
  .cfg_group(cfg_group),
  .cfg_mask(cfg_mask),
  .cfg_output_x(cfg_output_x),
  .cfg_source_width(cfg_source_width),
  .cfg_source_origin(cfg_source_origin),
  .cfg_storage_phase(cfg_storage_phase),.cfg_out_phase(cfg_out_phase),.cfg_row_banks(cfg_row_banks),
  .cfg_row_bases(cfg_row_bases),
  .out_banks(out_banks),
  .out_bases(out_bases),
  .cfg_weight_offset(cfg_weight_offset),
  .pcv(pcv),
  .pcr(pcr),
  .hv(hv),
  .hready(hready),
  .hbias(hbias),
  .hmult(hmult),
  .hrelu(hrelu),
  .hlayer(hlayer),
  .hgroups(hgroups),
  .hsvalid(hsvalid),
  .hssuccess(hssuccess),
  .hsready(hsready),
  .hbusy(hbusy),
  .herror(herror),
  .dma_valid(dvalid),
  .dma_ready(dready),
  .dstore(dstore),
  .drgb(drgb),
  .daddress(daddress),
  .dcolumns(dcolumns),
  .dgroups(dgroups),
  .dbank(dbank),
  .dbase(dbase),
  .ddone(ddone),
  .ddready(ddready),
  .derror(derror),
  .dbusy(dbusy),
  .busy(issue_busy),
  .frame_valid(issue_frame_valid),
  .error(issue_error),
  .fault_code(program_fault_code),
  .compute_completed(compute_completed),
  .dma_completed(dma_completed),
  .nodes_completed(nodes_completed),
  .rgb_pixels(rgb_pixels),
  .frame_cycles(production_cycles),
  .compute_begin(compute_begin),
  .compute_end(compute_end),
  .dma_end(dma_end),
  .debug_program_index(debug_program_index));
 // Speculation may read only structurally valid, bounded weight descriptors.
 // The issuer still performs normal full validation before execution.
 wire next_weight_ok=nv && ndata[511:504]==8'h55 && ndata[503:449]==0 &&
  (ndata[46+:3]==1 || ndata[46+:3]==2 || ndata[46+:3]==4) &&
  !(ndata[52] && ndata[54]) && (!ndata[54] || ndata[46+:3]==1) &&
  ndata[221+:3]==0 && ndata[221+:15]>=1792 &&
  {1'b0,ndata[221+:15]}+(ndata[54] ? 16'd216:ndata[52] ? {7'd0,ndata[46+:3],6'd0}:16'd72)<=16'd12608;
 // Validate the lookahead record once before exposing its narrow prefetch
 // request. It cannot be consumed until registered, and is revoked when the
 // current record retires (the fetcher then advances its lookahead slot).
 reg next_weight_qualified,next_weight_pw_q,next_weight_stem_q;
 reg [2:0] next_weight_groups_q;reg [14:0] next_weight_offset_q;
 always @(posedge clk)begin
  if(reset || cancel || discard_request || start_fire || cr || fetch_error)next_weight_qualified<=0;
  else begin
   next_weight_qualified<=next_weight_ok;
   if(next_weight_ok)begin
    next_weight_pw_q<=ndata[52];next_weight_stem_q<=ndata[54];
    next_weight_groups_q<=ndata[46+:3];next_weight_offset_q<=ndata[221+:15];
   end
  end
 end
 // synthesis translate_off
 always @(posedge clk)if(!reset && !cancel && next_weight_qualified && nv)begin
  if(!next_weight_ok || {next_weight_pw_q,next_weight_stem_q,next_weight_groups_q,next_weight_offset_q} !==
    {ndata[52],ndata[54],ndata[46+:3],ndata[221+:15]})
   $fatal(1,"NEXT_WEIGHT_LOOKAHEAD_OWNERSHIP");
 end
 // synthesis translate_on

 cnn_frame_datapath #(.NEXT_WEIGHT_PREFETCH(1),.AUTO_DW_GROUPS(1)) core(
  .next_weight_valid(next_weight_qualified && nv && !discard_request),.next_weight_pw(next_weight_pw_q),.next_weight_stem(next_weight_stem_q),
  .next_weight_groups(next_weight_groups_q),.next_weight_offset(next_weight_offset_q),
  .clk(clk),
  .reset(reset),
  .cancel(cancel || fetch_error || issue_error),
  .recover(1'b0),
  .recover_ready(dp_recover_ready),
  .quiesced(dp_quiesced),
  .halted(halted),
  .needs_common_reset(dp_needs_common_reset),
  .fault_code(fault_code),
  .cfg_valid(cfg_valid),
  .cfg_ready(cfg_ready),
  .done_valid(done_valid),
  .done_ready(done_ready),
  .busy(dp_busy),
  .error(dp_error),
  .cfg_pw_fuse(cfg_pw_fuse),
  .cfg_main_scaled(cfg_main_scaled),
  .cfg_stride2(cfg_stride2),.cfg_stride4(cfg_stride4),.cfg_nn4(cfg_nn4),
  .cfg_stem(cfg_stem),
  .cfg_mode_pw(cfg_mode_pw),
  .cfg_nn_view(cfg_nn_view),
  .cfg_skip_banks(cfg_skip_banks),
  .cfg_skip_bases(cfg_skip_bases),
  .cfg_skip_delta(cfg_skip_delta),
  .align_multiplier(align_multiplier),
  .post_multiplier(post_multiplier),
  .cfg_pixels(cfg_pixels),
  .cfg_rows(cfg_rows),
  .cfg_in_groups(cfg_in_groups),
  .cfg_groups(cfg_groups),
  .cfg_group(cfg_group),
  .cfg_mask(cfg_mask),
  .cfg_output_x(cfg_output_x),
  .cfg_source_width(cfg_source_width),
  .cfg_source_origin(cfg_source_origin),
  .cfg_storage_phase(cfg_storage_phase),.cfg_out_phase(cfg_out_phase),.cfg_row_banks(cfg_row_banks),
  .cfg_row_bases(cfg_row_bases),
  .out_banks(out_banks),
  .out_bases(out_bases),
  .cfg_weight_offset(cfg_weight_offset),
  .job_image_tag(job_image_tag),
  .pcv(pcv),
  .pcr(pcr),
  .hv(hv),
  .hready(hready),
  .hbias(hbias),
  .hmult(hmult),
  .hrelu(hrelu),
  .hlayer(hlayer),
  .hgroups(hgroups),
  .hsvalid(hsvalid),
  .hssuccess(hssuccess),
  .hsready(hsready),
  .hbusy(hbusy),
  .herror(herror),
  .begin_valid(begin_valid),
  .begin_ready(begin_ready),
  .begin_version(begin_version),
  .begin_image_bytes(begin_image_bytes),
  .begin_tag(begin_tag),
  .load_valid(load_valid),
  .load_ready(load_ready),
  .load_index(load_index),
  .load_data(load_data),
  .load_last(load_last),
  .verify_valid(verify_valid),
  .verify_ready(verify_ready),
  .verify_tag(verify_tag),
  .verify_ok(verify_ok),
  .diag_valid(diag_valid),
  .diag_accept(diag_accept),
  .diag_bank(diag_bank),
  .diag_address(diag_address),
  .diag_response(diag_response),
  .diag_ready(diag_ready),
  .diag_data(diag_data),
  .diag_error(diag_error),
  .diag_return_bank(diag_return_bank),
  .diag_return_address(diag_return_address),
  .diag_return_tag(diag_return_tag),
  .coeff_ready(coeff_ready),
  .loading(loading),
  .load_complete(load_complete),
  .store_error(store_error),
  .error_code(error_code),
  .image_tag(image_tag),
  .loaded_words(loaded_words),
  .dvalid(dvalid),
  .dready(dready),
  .dstore(dstore),
  .drgb(drgb),
  .daddress(daddress),
  .dcolumns(dcolumns),
  .dgroups(dgroups),
  .dbank(dbank),
  .dbase(dbase),
  .ddone(ddone),
  .ddready(ddready),
  .derror(derror),
  .dbusy(dbusy),
  .araddr(dp_araddr),
  .awaddr(awaddr),
  .arlen(dp_arlen),
  .awlen(awlen),
  .arvalid(dp_arvalid),
  .awvalid(awvalid),
  .rready(dp_rready),
  .wlast(wlast),
  .wvalid(wvalid),
  .bready(bready),
  .arready(dp_arready),
  .awready(awready),
  .wready(wready),
  .rvalid(dp_rvalid),
  .rlast(dp_rlast),
  .bvalid(bvalid),
  .rdata(dp_rdata),
  .rresp(dp_rresp),
  .bresp(bresp),
  .wdata(wdata),
  .wstrb(wstrb),
  .cycles(cycles),
  .reads(reads),
  .writes(writes),
  .acks(acks),
  .dcycles(dcycles),
  .dwords(dwords),
  .dbeats(dbeats),
  .arb_busy(arb_busy),
  .arena_router_busy(arena_router_busy));
 axi_read_merge2 read_merge(.clk(clk),.reset(reset),
  .s_araddr({fetch_araddr,dp_araddr}),.s_arlen({fetch_arlen,dp_arlen}),.s_arvalid({fetch_arvalid,dp_arvalid}),
  .s_arready(merged_arready),.s_rdata(merged_rdata),.s_rresp(merged_rresp),.s_rlast(merged_rlast),.s_rvalid(merged_rvalid),
  .s_rready({fetch_rready,dp_rready}),.m_araddr(araddr),.m_arlen(arlen),.m_arvalid(arvalid),.m_arready(arready),
  .m_rdata(rdata),.m_rresp(rresp),.m_rlast(rlast),.m_rvalid(rvalid),.m_rready(rready),.busy(merge_busy));
 assign {fetch_arready,dp_arready}=merged_arready;
 assign dp_rdata=merged_rdata[127:0];assign dp_rresp=merged_rresp[1:0];
 assign dp_rlast=merged_rlast[0];assign dp_rvalid=merged_rvalid[0];
endmodule
