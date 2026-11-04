// Synthesizable shared CNN data path. Commands are external handshakes; no
// testbench memory, responder, golden values or pressure generator is inside.
// Local abort revokes the shared on-chip arena synchronously for BOTH clients.
// DDR DMA never receives that local reset: accepted AXI transactions drain.
// reset is a COMMON transaction reset with the downstream AXI fabric/DDR user
// layer. An unresponsive downstream locks the block until that common reset.
module cnn_frame_datapath #(
 parameter integer WATCHDOG_CYCLES=1048576, NEXT_WEIGHT_PREFETCH=0, AUTO_DW_GROUPS=0
)(
 input wire clk,reset,cancel,recover,
 input wire next_weight_valid,next_weight_pw,next_weight_stem,
 input wire [2:0] next_weight_groups,input wire [14:0] next_weight_offset,
 input wire cfg_storage_phase,input wire [3:0] cfg_out_phase,
 output wire recover_ready,quiesced,halted,needs_common_reset,
 output reg [7:0] fault_code,
 input wire cfg_valid,output wire cfg_ready,output wire done_valid,input wire done_ready,
 output wire busy,error,
 input wire cfg_pw_fuse,cfg_main_scaled,cfg_stride2,cfg_stride4,cfg_nn4,cfg_stem,cfg_mode_pw,cfg_nn_view,
 input wire [11:0] cfg_skip_banks,input wire [39:0] cfg_skip_bases,input wire [10:0] cfg_skip_delta,
 input wire [17:0] align_multiplier,post_multiplier,
 input wire [8:0] cfg_pixels,input wire [2:0] cfg_rows,cfg_in_groups,cfg_groups,
 input wire [1:0] cfg_group,input wire [7:0] cfg_mask,
 input wire [10:0] cfg_output_x,cfg_source_width,cfg_source_origin,
 input wire [17:0] cfg_row_banks,input wire [59:0] cfg_row_bases,
 input wire [11:0] out_banks,input wire [39:0] out_bases,input wire [14:0] cfg_weight_offset,
 input wire [15:0] job_image_tag,
 input wire pcv,output wire pcr,
 input wire hv,output wire hready,input wire [14:0] hbias,hmult,hrelu,
 input wire [7:0] hlayer,input wire [2:0] hgroups,
 output wire hsvalid,hssuccess,input wire hsready,output wire hbusy,herror,
 input wire begin_valid,output wire begin_ready,input wire [31:0] begin_version,begin_image_bytes,
 input wire [15:0] begin_tag,
 input wire load_valid,output wire load_ready,input wire [11:0] load_index,
 input wire [63:0] load_data,input wire load_last,
 input wire verify_valid,output wire verify_ready,input wire [15:0] verify_tag,input wire verify_ok,
 input wire diag_valid,output wire diag_accept,input wire [2:0] diag_bank,input wire [9:0] diag_address,
 output wire diag_response,input wire diag_ready,output wire [63:0] diag_data,output wire [3:0] diag_error,
 output wire [2:0] diag_return_bank,output wire [9:0] diag_return_address,output wire [15:0] diag_return_tag,
 output wire coeff_ready,loading,load_complete,store_error,output wire [3:0] error_code,
 output wire [15:0] image_tag,output wire [11:0] loaded_words,
 input wire dvalid,output wire dready,input wire dstore,drgb,input wire [31:0] daddress,
 input wire [8:0] dcolumns,input wire [2:0] dgroups,input wire [2:0] dbank,input wire [9:0] dbase,
 output wire ddone,input wire ddready,output wire derror,dbusy,
 output wire [31:0] araddr,awaddr,output wire [7:0] arlen,awlen,
 output wire arvalid,awvalid,rready,wlast,wvalid,bready,
 input wire arready,awready,wready,rvalid,rlast,bvalid,
 input wire [127:0] rdata,input wire [1:0] rresp,bresp,
 output wire [127:0] wdata,output wire [15:0] wstrb,
 output wire [31:0] cycles,reads,writes,acks,dcycles,dwords,dbeats,
 output wire arb_busy,arena_router_busy
);
 wire hfv,hfr,huv,hur;wire [1:0] hfg,hug;
 wire [255:0] cached_bias,cached_multiplier;wire [7:0] cached_relu;
 wire header_error;
 wire header_apply=pcv && pcr;
 assign herror=header_error;
 wire core_error,core_done,core_cfg_ready,core_pcr,header_ready,header_status,dma_ready,dma_done;
 wire arb_error,arena_router_error,flush;
 reg stopping,stopped,timed_out,header_loaded;
 localparam integer WATCH_WIDTH=$clog2(WATCHDOG_CYCLES+1);
 wire progress,work_active,recover_fire,local_reset;
 wire [13:0] progress_events;wire watchdog_expired;
 fixed_progress_watchdog #(.CYCLES(WATCHDOG_CYCLES),.EVENTS(14)) watchdog_monitor(
  .clk(clk),.reset(local_reset),.work_active(work_active),.progress_events(progress_events),.expired(watchdog_expired));
 // CNN input/output/feature windows, each exactly 4MiB. Extents include all
 // logical bytes; DMA separately enforces aligned beat/4KiB and arena bounds.
 wire [15:0] dma_bytes=drgb ? ({7'd0,dcolumns}<<1)+{7'd0,dcolumns}:
  dgroups==4 ? {2'd0,dcolumns,5'd0}:dgroups==2 ? {3'd0,dcolumns,4'd0}:{4'd0,dcolumns,3'd0};
 wire window_ok;
 fixed_dma_window_guard guard(.address(daddress),.bytes(dma_bytes),.ok(window_ok));
 wire request_bad=dvalid && !window_ok;
 // Reject an illegal command immediately at dready/cmd_valid. Latch its
 // fault separately so descriptor decode never drives every in-flight
 // client's combinational cancel/READY path. No illegal AXI request escapes.
 reg address_fault;
 always @(posedge clk)begin
  if(reset || recover_fire)address_fault<=0;
  else if(request_bad)address_fault<=1;
 end
 wire [7:0] causes={watchdog_expired,address_fault,store_error,arena_router_error,arb_error,herror,derror,core_error};
 wire abort_pulse=!stopping && !stopped && (cancel || (|causes));
 // Each client rejects its own bad operation immediately. Global teardown
 // crosses a registered fault boundary: a combinational busy/watchdog/error
 // cone must not feed back through every client's cancel and READY paths.
 // External cancellation and the registered illegal-address latch stay direct.
 assign halted=stopping || stopped || cancel || address_fault;
 reg abort_registered;
 always @(posedge clk)begin
  if(reset || recover_fire)abort_registered<=0;
  else abort_registered<=abort_pulse;
 end
 // The compute owner drains ABORT only after cancel is released. Preserve
 // its pulse contract; arena/DMA separately retain the level stop barrier.
 wire abort_cancel=abort_registered || (!stopping && !stopped && cancel);
 wire arena_flush=halted || flush;
 assign quiesced=halted && !busy && !dbusy && !hbusy && !arb_busy && !arena_router_busy;
 assign recover_ready=stopped && quiesced && !cancel;
 assign recover_fire=recover && recover_ready;
 assign local_reset=reset || recover_fire;
 assign needs_common_reset=timed_out && !quiesced;
 assign error=(|causes) || halted;
 // A host coefficient transaction wins a simultaneous idle-cycle request.
 // Keep ready and valid gates identical; otherwise both clients can believe
 // that incompatible commands were accepted on the same edge.
 wire host_request=begin_valid || load_valid || verify_valid || diag_valid;
 wire compute_enable=!halted && !host_request && !loading && !diag_response;
 assign cfg_ready=core_cfg_ready && compute_enable;
 assign done_valid=core_done && !error;
 assign pcr=core_pcr && header_loaded && compute_enable;
 assign hready=header_ready && compute_enable && !pcv;
 assign hsvalid=header_status && !error;
 assign dready=dma_ready && !halted && window_ok;
 assign ddone=dma_done && !error;
 wire host_idle=!busy && !dbusy && !hbusy && !arb_busy && !halted;
 wire begin_ready_i,load_ready_i,verify_ready_i,diag_accept_i;
 assign begin_ready=begin_ready_i && host_idle;
 assign load_ready=load_ready_i && host_idle;
 assign verify_ready=verify_ready_i && host_idle;
 assign diag_accept=diag_accept_i && host_idle;

 wire [4:0] qv,qr,svw,srw,store_qr,store_rv,store_rr,rq_valid;
 wire [44:0] qa,sa,rq_address;wire [79:0] qt,stg;wire [319:0] sdw;wire [19:0] se;
 // The coefficient store's fast port now serves only the weight loader.
 // Header parameters are mirrored during host load; no inference-side arbiter.
 assign rq_valid=qv;assign rq_address=qa;assign qr=store_qr;
 assign svw=store_rv;assign store_rr=srw;
 // Quiescence observes registered accepted tokens, never a reset-gated
 // request/response VALID (which would feed recover_ready back into reset).
 reg [4:0] coefficient_tokens;
 always @(posedge clk)begin
  if(local_reset)coefficient_tokens<=0;
  else coefficient_tokens<=(coefficient_tokens & ~(store_rv & store_rr)) | (qv & store_qr);
 end
 assign arb_busy=|coefficient_tokens;assign arb_error=1'b0;
 // Existing measurement taps remain explicit: no header-side reads occur.
 wire [4:0] hqv=5'd0,hqr=5'd0;
 coefficient_bank_store_fast store(.clk(clk),.reset(reset),.cancel(recover_fire),
  .begin_valid(begin_valid && host_idle),.begin_ready(begin_ready_i),.begin_version(begin_version),.begin_image_bytes(begin_image_bytes),.begin_tag(begin_tag),
  .load_valid(load_valid && host_idle),.load_ready(load_ready_i),.load_index(load_index),.load_data(load_data),.load_last(load_last),
  .verify_valid(verify_valid && host_idle),.verify_ready(verify_ready_i),.verify_tag(verify_tag),.verify_ok(verify_ok),
  .read_valid(diag_valid && host_idle),.read_ready(diag_accept_i),.read_for_compute(1'b0),.read_bank(diag_bank),.read_address(diag_address),
  .response_valid(diag_response),.response_ready(diag_ready),.response_data(diag_data),.response_error(diag_error),
  .response_bank(diag_return_bank),.response_address(diag_return_address),.response_tag(diag_return_tag),
  .loading(loading),.load_complete(load_complete),.coeff_ready(coeff_ready),.error_valid(store_error),.error_code(error_code),.image_tag(image_tag),.loaded_words(loaded_words),
  .fast_read_valid(rq_valid),.fast_read_ready(store_qr),.fast_read_address(rq_address),.fast_read_tag(job_image_tag),
  .fast_response_valid(store_rv),.fast_response_ready(store_rr),.fast_response_data(sdw),.fast_response_address(sa),.fast_response_tag(stg),.fast_response_error(se));
 wire [1023:0] pcb={768'd0,cached_bias},pcm={768'd0,cached_multiplier};wire [31:0] pcrelu={24'd0,cached_relu};
 resident_header_table header(.clk(clk),.reset(local_reset),.cancel(abort_cancel),
  .image_write(load_valid && load_ready),.image_index(load_index),.image_data(load_data),
  .load_cmd_valid(hv && compute_enable && !pcv),.load_cmd_ready(header_ready),
  .load_bias_offset(hbias),.load_multiplier_offset(hmult),.load_relu_offset(hrelu),
  .load_groups(hgroups),.load_image_tag(job_image_tag),
  .load_status_valid(header_status),.load_status_ready(hsready || halted),.load_status_success(hssuccess),
  .store_coeff_ready(coeff_ready),.store_image_tag(image_tag),.apply(header_apply),
  .first_valid(hfv),.first_group(hfg),.first_ready(hfr),.use_valid(huv),.use_group(hug),.use_ready(hur),
  .bias(cached_bias),.multiplier(cached_multiplier),.relu(cached_relu),.busy(hbusy),.error(header_error));
 wire [5:0] av,ar,sv,sr,tv,dma_ar,dma_sv,dsr,dma_sw;
 wire [59:0] aa,ta;wire [767:0] sd,td,dma_sd;wire [11:0] th;
 wire [5:0] bv,br,bs,bsr,bsw;wire [59:0] ba;wire [11:0] bh;wire [767:0] bd,bsd;
 wire [5:0] aux_write_collision;
 row_pair_arena6 #(.EXTERNAL_WRITE_COLLISION(1)) arena(.clk(clk),.reset(local_reset),.cancel(arena_flush),
  .read_valid(av),.read_ready(ar),.read_address(aa),.read_response(sv),.read_response_ready(sr),.read_data(sd),
  .aux_write_collision(aux_write_collision),.aux_valid(bv),.aux_ready(br),.aux_address(ba),.aux_write_half(bh),.aux_data(bd),
  .aux_response(bs),.aux_response_ready(bsr),.aux_response_data(bsd),.aux_was_write(bsw));
 row_pair_compute_cluster #(.EXTERNAL_HEADER(1),.NEXT_WEIGHT_PREFETCH(NEXT_WEIGHT_PREFETCH),.AUTO_DW_GROUPS(AUTO_DW_GROUPS)) dut(
  .next_weight_valid(next_weight_valid && compute_enable),.next_weight_pw(next_weight_pw),.next_weight_stem(next_weight_stem),
  .next_weight_groups(next_weight_groups),.next_weight_offset(next_weight_offset),
  .header_first_valid(hfv),.header_first_group(hfg),.header_first_ready(hfr),
  .header_use_valid(huv),.header_use_group(hug),.header_use_ready(hur),.clk(clk),.reset(local_reset),.cancel(abort_cancel),
  .cfg_valid(cfg_valid && compute_enable),.cfg_ready(core_cfg_ready),.done_valid(core_done),.done_ready(done_ready || halted),.busy(busy),.error(core_error),
  .cfg_pw_fuse(cfg_pw_fuse),.cfg_main_scaled(cfg_main_scaled),.cfg_skip_banks(cfg_skip_banks),.cfg_skip_bases(cfg_skip_bases),.cfg_skip_delta(cfg_skip_delta),
  .cfg_stride2(cfg_stride2),.cfg_stride4(cfg_stride4),.cfg_nn4(cfg_nn4),.cfg_stem(cfg_stem),.cfg_residual(1'b0),.cfg_row_deltas(66'd0),.pc_residual(1'b0),.pc_align_multiplier(align_multiplier),.pc_post_multiplier(post_multiplier),
  .cfg_mode_pw(cfg_mode_pw),.cfg_nn_view(cfg_nn_view),.cfg_pixels(cfg_pixels),.cfg_rows(cfg_rows),.cfg_in_groups(cfg_in_groups),.cfg_out_groups(cfg_groups),
  .cfg_group(cfg_group),.cfg_mask(cfg_mask),.cfg_output_x(cfg_output_x),.cfg_source_width(cfg_source_width),.cfg_source_origin(cfg_source_origin),
  .cfg_storage_phase(cfg_storage_phase),.cfg_out_phase(cfg_out_phase),.cfg_row_banks(cfg_row_banks),.cfg_row_bases(cfg_row_bases),.cfg_out_banks(out_banks),.cfg_out_bases(out_bases),
  .cfg_weight_offset(cfg_weight_offset),.cfg_image_tag(job_image_tag),.pcv(pcv && header_loaded && compute_enable),.pcr(core_pcr),.pcb(pcb),.pcm(pcm),.pcrelu(pcrelu),
  .store_coeff_ready(coeff_ready),.store_image_tag(image_tag),.fast_read_valid(qv),.fast_read_ready(qr),.fast_read_address(qa),.fast_read_tag(qt),
  .fast_response_valid(svw),.fast_response_ready(srw),.fast_response_data(sdw),.fast_response_address(sa),.fast_response_tag(stg),.fast_response_error(se),
  .flush_arena(flush),
  .read_valid(av),.read_ready(ar),.read_address(aa),.read_response(sv),.read_response_ready(sr),.read_data(sd),
  .dma_req_valid(tv),.dma_req_ready(dma_ar),.dma_req_address(ta),.dma_req_half(th),.dma_req_data(td),
  .dma_rsp_valid(dma_sv),.dma_rsp_ready(dsr),.dma_rsp_data(dma_sd),.dma_rsp_write(dma_sw),
  .aux_write_collision(aux_write_collision),.aux_valid(bv),.aux_ready(br),.aux_address(ba),.aux_write_half(bh),.aux_data(bd),
  .aux_response(bs),.aux_response_ready(bsr),.aux_response_data(bsd),.aux_was_write(bsw),
  .fabric_busy(arena_router_busy),.fabric_error(arena_router_error),
  .cycles(cycles),.reads(reads),.writes(writes),.acks(acks));
 row_pair_dma dma(.clk(clk),.reset(local_reset),.cancel(arena_flush),
  .cmd_valid(dvalid && !halted && window_ok),.cmd_ready(dma_ready),.cmd_store(dstore),.cmd_rgb(drgb),
  .cmd_address(daddress),.cmd_columns(dcolumns),.cmd_groups(dgroups),.cmd_bank(dbank),.cmd_phase(!drgb && daddress[5]),.cmd_base(dbase),
  .done_valid(dma_done),.done_ready(ddready || halted),.error(derror),.busy(dbusy),
  .av(tv),.ar(dma_ar),.ah(th),.aa(ta),.ad(td),.sv(dma_sv),.sr(dsr),.sd(dma_sd),.sw(dma_sw),
  .araddr(araddr),.arlen(arlen),.arvalid(arvalid),.arready(arready),.rdata(rdata),.rresp(rresp),.rlast(rlast),.rvalid(rvalid),.rready(rready),
  .awaddr(awaddr),.awlen(awlen),.awvalid(awvalid),.awready(awready),.wdata(wdata),.wstrb(wstrb),.wlast(wlast),.wvalid(wvalid),.wready(wready),
  .bresp(bresp),.bvalid(bvalid),.bready(bready),.cycles(dcycles),.arena_words(dwords),.axi_beats(dbeats));
 assign work_active=busy || dbusy || hbusy;
 assign progress_events={ (|(bv&br)), (|(bs&bsr)), (|(av&ar)), (|(sv&sr)), (|(rq_valid&store_qr)), (|(store_rv&store_rr)),
  (arvalid&&arready), (awvalid&&awready), (rvalid&&rready), (wvalid&&wready), (bvalid&&bready),
  (core_done&&(done_ready||halted)), (dma_done&&(ddready||halted)), (header_status&&(hsready||halted))};
 assign progress=|progress_events;
 always @(posedge clk)begin
  if(local_reset)begin stopping<=0;stopped<=0;timed_out<=0;fault_code<=0;header_loaded<=0;end
  else begin
   if(watchdog_expired)timed_out<=1;
   if(request_bad && !stopping && !stopped)fault_code<=causes | 8'h40;
   if(abort_pulse)begin stopping<=1;fault_code<=causes;header_loaded<=0;end
   if(stopping && quiesced)begin stopping<=0;stopped<=1;end
   if(hv && hready)header_loaded<=0;
   if(begin_valid && begin_ready)header_loaded<=0;
   if(header_status && hsready && hssuccess && !halted)header_loaded<=1;
  end
 end
endmodule

// Fixed coefficient-image v1 header table. Load once with the unchanged host
// stream; the existing coefficient store still owns order, CRC/verify and tag.
// Eight 20-bit parameter RAMs retain all 76 groups: low/high bias and scale
// share one physical lane RAM. Three read clocks replace sparse parallel RAMs.
// One asymmetric RAM receives each ReLU word over four startup-only writes.
module resident_header_table(
 input wire clk,reset,cancel,
 input wire image_write,input wire [11:0] image_index,input wire [63:0] image_data,
 input wire load_cmd_valid,output wire load_cmd_ready,
 input wire [14:0] load_bias_offset,load_multiplier_offset,load_relu_offset,
 input wire [2:0] load_groups,input wire [15:0] load_image_tag,
 input wire store_coeff_ready,input wire [15:0] store_image_tag,
 output wire load_status_valid,input wire load_status_ready,output wire load_status_success,
 input wire apply,input wire first_valid,input wire [1:0] first_group,output wire first_ready,
 input wire use_valid,input wire [1:0] use_group,output wire use_ready,
 output wire [255:0] bias,multiplier,output wire [7:0] relu,
 output wire busy,output reg error
);
 reg status,success,prepared,configured,have;
 reg [6:0] prepared_bias,prepared_scale,prepared_relu,active_bias,active_scale,active_relu;
 reg [2:0] prepared_groups,active_groups;reg [15:0] prepared_tag;
 reg [1:0] held_group,mask_part,fetch_state;
 reg bias_pending;reg [1:0] bias_pair;reg [6:0] bias_group;reg [63:0] bias_word_saved;reg mask_pending;reg [63:0] mask_word;reg [3:0] mask_address;
 wire flush=reset || cancel;
 wire revoked=!store_coeff_ready || store_image_tag!=prepared_tag;
 wire [15:0] field_bytes={8'd0,load_groups,5'd0};
 wire good=(load_groups==1 || load_groups==2 || load_groups==4) &&
  load_bias_offset[4:0]==0 && load_multiplier_offset[4:0]==0 &&
  load_bias_offset>=12608 && {1'b0,load_bias_offset}+field_bytes<=15040 &&
  load_multiplier_offset>=15040 && {1'b0,load_multiplier_offset}+field_bytes<=17472 &&
  load_relu_offset>=17472 && {1'b0,load_relu_offset}+{13'd0,load_groups}<=17600 &&
  store_coeff_ready && load_image_tag==store_image_tag;
 assign load_cmd_ready=!flush && !error && !status && !mask_pending && !bias_pending;
 assign load_status_valid=!flush && status && (!success || !revoked);
 assign load_status_success=success && !revoked;
 assign busy=status || mask_pending || bias_pending || fetch_state!=0;
 wire hit=have && held_group==use_group;
 wire same_view=configured && have && held_group==0 && fetch_state==0 && active_bias==prepared_bias &&
  active_scale==prepared_scale && active_relu==prepared_relu && active_groups==prepared_groups;
 // Production APPLY begins at group zero. Fetch it on the APPLY edge,
 // before CFG acceptance; stale last-group selection must not start a detour.
 wire apply_read=apply && prepared && !revoked && !status && !flush && !error && !same_view;
 wire start_read=configured && store_coeff_ready && !flush && !error && !apply && !hit && fetch_state==0;
 wire re=apply_read || start_read || (configured && store_coeff_ready && !flush && !error && !apply && (fetch_state==1 || fetch_state==2));
 // APPLY is admitted only at the compute owner's idle boundary. Consumers
 // cannot use either ready on that edge; register updates define the next
 // header view. Keep all RAM APPLY/read exclusion below unchanged.
 assign first_ready=configured && store_coeff_ready && !flush && !error;
 // synthesis translate_off
 always @(posedge clk)if(!flush && apply && (first_valid || use_valid))
  $fatal(1,"HEADER_APPLY_CONSUMER_NOT_IDLE");
 // synthesis translate_on
 assign use_ready=first_ready && hit && {1'b0,use_group}<active_groups;
 wire wb=image_write && !flush && image_index>=1576 && image_index<1880;
 wire ws=image_write && !flush && image_index>=1880 && image_index<2184;
 wire wr=image_write && !flush && image_index>=2184 && image_index<2200;
 wire [8:0] bias_word=9'(image_index-12'd1576),scale_word=9'(image_index-12'd1880);
 wire [3:0] relu_word=4'(image_index-12'd2184);
 wire [1:0] target_group=start_read ? use_group:held_group;
 wire [6:0] rb=apply_read ? prepared_bias:active_bias+{5'd0,target_group};
 wire [6:0] rs=active_scale+{5'd0,target_group};
 wire [6:0] rr=apply_read ? prepared_relu:active_relu+{5'd0,use_group};
 wire [8:0] ra=apply_read ? {1'b0,prepared_bias,1'b0}:fetch_state==2 ? {2'b10,rs}:{1'b0,rb,fetch_state==1};
 for(genvar lane=0;lane<8;lane=lane+1)begin:LANE
  wire [31:0] field=image_data[(lane%2)*32+:32];
  wire bad_scale=field[31:18]!={14{field[17]}};
  wire eb=wb && bias_word[1:0]==2'(lane/2),es=ws && scale_word[1:0]==2'(lane/2);
  wire wh=bias_pending && bias_pair==2'(lane/2) && !flush;
  wire [31:0] old_field=bias_word_saved[(lane%2)*32+:32];
  wire [8:0] wa=wh ? {1'b0,bias_group,1'b1}:es ? {2'b10,scale_word[8:2]}:{1'b0,bias_word[8:2],1'b0};
  wire [19:0] wd=wh ? {8'd0,old_field[31:20]}:es ? {1'b0,bad_scale,field[17:0]}:field[19:0];
  wire [19:0] q;
  reg [31:0] held_bias;
  EFX_RAM10 #(.READ_WIDTH(20),.WRITE_WIDTH(20),.OUTPUT_REG(0),.WRITE_MODE("READ_FIRST")) parameters_ram(
   .WCLK(clk),.WE({2{eb || es || wh}}),.WCLKE(1'b1),.WADDREN(1'b1),.WDATA(wd),.WADDR(wa),
   .RCLK(clk),.RE(re),.RST(1'b0),.RADDREN(1'b1),.RDATA(q),.RADDR(ra));
  // Payload never resets; only a completed three-read token exposes it.
  always @(posedge clk)if(!flush && !apply)begin
   if(fetch_state==1)held_bias[19:0]<=q;
   if(fetch_state==2)held_bias[31:20]<=q[11:0];
  end
  assign bias[lane*32+:32]=held_bias;
  assign multiplier[lane*32+:32]={{13{q[17]}},q[17]^q[18],q[17:0]};
 end
 EFX_RAM10 #(.READ_WIDTH(8),.WRITE_WIDTH(16),.OUTPUT_REG(0),.WRITE_MODE("READ_FIRST")) relu_masks(
  .WCLK(clk),.WE({2{mask_pending && !flush}}),.WCLKE(1'b1),.WADDREN(1'b1),
  .WDATA(mask_word[mask_part*16+:16]),.WADDR({3'd0,mask_address,mask_part}),
  .RCLK(clk),.RE(start_read || apply_read),.RST(1'b0),.RADDREN(1'b1),.RDATA(relu),.RADDR({3'd0,rr}));
 always @(posedge clk)begin
  if(reset)begin
   status<=0;success<=0;prepared<=0;configured<=0;have<=0;error<=0;
   prepared_bias<=0;prepared_scale<=0;prepared_relu<=0;active_bias<=0;active_scale<=0;active_relu<=0;
   prepared_groups<=0;active_groups<=0;prepared_tag<=0;held_group<=0;fetch_state<=0;bias_pending<=0;bias_pair<=0;bias_group<=0;
   mask_pending<=0;mask_part<=0;mask_address<=0;
  end else if(cancel)begin status<=0;success<=0;prepared<=0;configured<=0;have<=0;mask_pending<=0;bias_pending<=0;fetch_state<=0;end
  else begin
   if(bias_pending)bias_pending<=0;
   if(wb)begin bias_pending<=1;bias_pair<=bias_word[1:0];bias_group<=bias_word[8:2];bias_word_saved<=image_data;end
   if(mask_pending)begin
    if(mask_part==3)mask_pending<=0;else mask_part<=mask_part+2'd1;
   end
   if(wr)begin mask_pending<=1;mask_part<=0;mask_address<=relu_word;mask_word<=image_data;end
   if(load_status_valid && load_status_ready)status<=0;
   if(status && revoked)begin success<=0;prepared<=0;error<=1;end
   if(!store_coeff_ready)begin prepared<=0;configured<=0;have<=0;fetch_state<=0;end
   if(load_cmd_valid && load_cmd_ready)begin
    status<=1;success<=good;prepared<=good;error<=!good;
    prepared_bias<=7'((load_bias_offset-15'd12608)>>5);
    prepared_scale<=7'((load_multiplier_offset-15'd15040)>>5);
    prepared_relu<=7'(load_relu_offset-15'd17472);prepared_groups<=load_groups;prepared_tag<=load_image_tag;
   end
   if(start_read)begin have<=0;held_group<=use_group;fetch_state<=1;end
   else if(fetch_state!=0 && store_coeff_ready)begin
    if(fetch_state==2)begin have<=1;fetch_state<=0;end
    else fetch_state<=fetch_state+1'b1;
   end
   if(apply)begin
    if(!prepared || revoked || status)error<=1;
    else begin
     configured<=1;active_bias<=prepared_bias;active_scale<=prepared_scale;active_relu<=prepared_relu;
     active_groups<=prepared_groups;have<=same_view;held_group<=0;fetch_state<=apply_read ? 2'd1:2'd0;
    end
   end
   if(use_valid && !use_ready)error<=1;
   if(image_write && (store_coeff_ready || mask_pending || bias_pending))error<=1;
  end
 end
endmodule

// Exact four aligned 4MiB-window check. Each legal transfer is <64KiB.
// Carry into bit 22 can only occur through address[21:16]==63. Reaching
// the exclusive upper bound exactly is legal; crossing it is rejected.
module fixed_dma_window_guard(
 input wire [31:0] address,input wire [15:0] bytes,output wire ok
);
 wire [16:0] low_sum={1'b0,address[15:0]}+{1'b0,bytes};
 wire crosses=(address[21:16]==6'h3f) && low_sum[16] && (|low_sum[15:0]);
 wire region=address[31:22]==10'd0 || address[31:22]==10'd2 ||
  address[31:22]==10'd8 || address[31:22]==10'd12;
 assign ok=region && !crosses;
endmodule

// Retimed progress counter with exactly the original expiration edge.
// count is the old counter delayed by one edge. The prior event determines
// whether that counter increments or clears, so expire at CYCLES-2 only
// when the prior cycle was active and made no progress. No timeout extension.
module fixed_progress_watchdog #(parameter integer CYCLES=1048576,EVENTS=14)(
 input wire clk,reset,work_active,input wire [EVENTS-1:0] progress_events,output wire expired
);
 localparam integer WIDTH=$clog2(CYCLES+1);
 reg [WIDTH-1:0] count;reg work_q;reg [EVENTS-1:0] events_q;
 wire prior_step=work_q && !(|events_q);
 assign expired=CYCLES==1 ? work_active:
  work_active && prior_step && count>=WIDTH'(CYCLES-2);
 always @(posedge clk)begin
  if(reset)begin count<=0;work_q<=0;events_q<=0;end
  else begin
   work_q<=work_active;events_q<=progress_events;
   if(!prior_step)count<=0;
   else if(count<WIDTH'(CYCLES-1))count<=count+1'b1;
  end
 end
endmodule
