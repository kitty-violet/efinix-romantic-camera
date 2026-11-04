// Fixed production adapter; preserves public descriptor/coefficient/ACK interfaces.
module row_pair_compute_cluster #(parameter integer EXTERNAL_HEADER=0, NEXT_WEIGHT_PREFETCH=0, AUTO_DW_GROUPS=0)(
 input wire clk,reset,cancel,
 input wire next_weight_valid,next_weight_pw,next_weight_stem,
 input wire [2:0] next_weight_groups,input wire [14:0] next_weight_offset,
 output wire header_first_valid,output wire [1:0] header_first_group,input wire header_first_ready,
 output wire header_use_valid,output wire [1:0] header_use_group,input wire header_use_ready,
 input wire cfg_valid,output wire cfg_ready,output wire done_valid,input wire done_ready,output wire busy,output reg error,
 input wire cfg_pw_fuse,cfg_main_scaled,input wire [11:0] cfg_skip_banks,input wire [39:0] cfg_skip_bases,input wire [10:0] cfg_skip_delta,
 input wire cfg_stride2,cfg_stem,cfg_stride4,cfg_nn4,input wire cfg_residual,input wire [65:0] cfg_row_deltas,
 input wire pc_residual,input wire [17:0] pc_align_multiplier,pc_post_multiplier,
 input wire cfg_mode_pw,cfg_nn_view,input wire [8:0] cfg_pixels,input wire [2:0] cfg_rows,cfg_in_groups,cfg_out_groups,
 input wire [1:0] cfg_group,input wire [7:0] cfg_mask,
 input wire [10:0] cfg_output_x,cfg_source_width,cfg_source_origin,
 input wire cfg_storage_phase,input wire [3:0] cfg_out_phase,input wire [17:0] cfg_row_banks,input wire [59:0] cfg_row_bases,
 input wire [11:0] cfg_out_banks,input wire [39:0] cfg_out_bases,
 input wire [14:0] cfg_weight_offset,input wire [15:0] cfg_image_tag,
 input wire pcv,output wire pcr,input wire [1023:0] pcb,pcm,input wire [31:0] pcrelu,
 input wire store_coeff_ready,input wire [15:0] store_image_tag,
 output wire [4:0] fast_read_valid,input wire [4:0] fast_read_ready,
 output wire [44:0] fast_read_address,output wire [79:0] fast_read_tag,
 input wire [4:0] fast_response_valid,output wire [4:0] fast_response_ready,
 input wire [319:0] fast_response_data,input wire [44:0] fast_response_address,
 input wire [79:0] fast_response_tag,input wire [19:0] fast_response_error,
 output wire flush_arena,
 output wire [5:0] read_valid,input wire [5:0] read_ready,output wire [59:0] read_address,
 input wire [5:0] read_response,output wire [5:0] read_response_ready,input wire [767:0] read_data,
 input wire [5:0] dma_req_valid,output wire [5:0] dma_req_ready,input wire [59:0] dma_req_address,
 input wire [11:0] dma_req_half,input wire [767:0] dma_req_data,
 output wire [5:0] dma_rsp_valid,input wire [5:0] dma_rsp_ready,output wire [767:0] dma_rsp_data,output wire [5:0] dma_rsp_write,
 output wire [5:0] aux_write_collision,
 output wire [5:0] aux_valid,input wire [5:0] aux_ready,output wire [59:0] aux_address,
 output wire [11:0] aux_write_half,output wire [767:0] aux_data,
 input wire [5:0] aux_response,output wire [5:0] aux_response_ready,input wire [767:0] aux_response_data,input wire [5:0] aux_was_write,
 output wire fabric_error,fabric_busy,
 output reg [31:0] cycles,output wire [31:0] reads,writes,acks
);
 localparam IDLE=0,HEADER=1,LOAD_START=2,LOAD_WAIT=3,COMPUTE_START=4,COMPUTE_RUN=5,ABORT=6,DONE=7;
 reg [3:0] state;reg loaded;reg preload_started,preload_done;
 reg mode,stem,stride2,nn_view,stride4,nn4,fused,main_scaled,storage_phase,skip_offset_two;
 reg [8:0] pixels;reg [2:0] rows,gi,go;reg [1:0] group_q,source_group;
 reg [7:0] result_mask;reg [10:0] ox,width,origin;
 reg [17:0] row_banks;reg [59:0] row_bases;reg [11:0] out_banks,skip_banks;
 reg [39:0] out_bases;reg [43:0] skip_bases;reg [3:0] out_phase;reg [14:0] weight_offset;reg [15:0] image;
 reg [17:0] align_scale,post_scale;
 reg [31:0] read_count;
 // These totals reset on each descriptor and retire at most four groups.
 // Even the full 13-bit child counter range gives 4*8191=32764, so
 // sixteen bits preserve every possible retired total, including bad jobs.
 reg [15:0] write_base,ack_base;
 wire [255:0] selected_bias,selected_multiplier;wire [7:0] selected_relu;
 generate if(EXTERNAL_HEADER!=0)begin:EXTERNAL
  assign selected_bias=pcb[255:0];assign selected_multiplier=pcm[255:0];assign selected_relu=pcrelu[7:0];
 end else begin:INTERNAL
  reg [1023:0] biases,multipliers;reg [31:0] relus;
  always @(posedge clk)if(pcv && pcr)begin biases<=pcb;multipliers<=pcm;relus<=pcrelu;end
  assign selected_bias=biases[group_q*256+:256];assign selected_multiplier=multipliers[group_q*256+:256];assign selected_relu=relus[group_q*8+:8];
 end endgenerate
 wire load_ready,load_done,load_busy,load_error;wire finish_now;
 wire video_ready,video_done,video_busy,video_error;
 wire [12:0] video_writes,video_acks;
 wire abort_event=state!=IDLE && state!=ABORT && state!=DONE &&
  (cancel || !store_coeff_ready || store_image_tag!=image ||
   (load_done && load_error) || (state==COMPUTE_RUN && (video_error || fabric_error)));
 // Detect/reject completion immediately at this owner; cancel the children
 // on the registered boundary, so coefficient/reset qualification cannot
 // form a combinational feedback path through RAM readiness and DMA faults.
 reg abort_registered;
 always @(posedge clk)if(reset)abort_registered<=0;else abort_registered<=abort_event;
 wire local_cancel=cancel || abort_registered;
 // Current result retirement and speculative coefficient ownership differ.
 // A retired result may release dependent DMA while the loader remains busy.
 // The aggregate busy/pcr barrier still prevents reconfiguration and stop ACK.
 assign busy=state!=IDLE || load_busy;assign done_valid=!reset && !cancel && (state==DONE || finish_now);
 assign pcr=!reset && !cancel && state==IDLE && !load_busy && !video_busy;
 assign cfg_ready=pcr && loaded && !pcv;
 wire next_group=!stem && {1'b0,group_q}+3'd1<go;
 wire chain_group=(mode || AUTO_DW_GROUPS!=0) && next_group;
 // Intermediate groups reuse the current descriptor. Only its final group
 // preloads the actual next descriptor, already held by the finite fetcher.
 // A next-header apply is separate: never prefetch its first-group parameters
 // through the still-active current header bank.
 wire next_job=NEXT_WEIGHT_PREFETCH!=0 && !next_group && next_weight_valid;
 wire can_prefetch=next_group || next_job;
 wire prefetch=state==COMPUTE_RUN && can_prefetch;
 wire [1:0] prefetched_group=next_group ? group_q+2'd1:2'd0;
 wire load_pw=prefetch && next_job ? next_weight_pw:mode;
 wire load_stem=prefetch && next_job ? next_weight_stem:stem;
 wire [2:0] load_groups=prefetch && next_job ? next_weight_groups:gi;
 wire [14:0] load_offset=prefetch ? (next_job ? next_weight_offset:
   mode ? weight_offset:weight_offset+15'd72):weight_offset;
 wire [1:0] load_group=prefetch ? prefetched_group:group_q;
 assign header_first_valid=EXTERNAL_HEADER!=0 && (state==HEADER || (prefetch && next_group)) && !local_cancel && !abort_event;
 assign header_first_group=prefetch && next_group ? prefetched_group:group_q;
 // Select a new descriptor while IDLE; never feed combinational video_done
 // into parameter ready/start/backend reset, which creates a structural loop.
 // MAC DONE is registered after its final RQ input handshake. Remaining
 // feedback operations use only align/post scales, not the header group.
 // The synchronous header read can overlap the remaining RQ/write ACK drain.
 wire engine_mac_done;
 // This is only a speculative header-read hint, never a completion or
 // ownership token. Register it to keep MAC/reset/READY logic out of the
 // next header comparison and coefficient-loader handshake. Clear whenever
 // leaving RUN so a previous group's DONE cannot prefetch a new group early.
 reg engine_mac_done_q;
 always @(posedge clk)begin
  if(reset || local_cancel || state!=COMPUTE_RUN)engine_mac_done_q<=0;
  else engine_mac_done_q<=engine_mac_done;
 end
 wire early_header=EXTERNAL_HEADER!=0 && state==COMPUTE_RUN && chain_group && engine_mac_done_q;
 assign header_use_group=early_header ? group_q+2'd1:state==IDLE ? (cfg_mode_pw || cfg_stem ? 2'd0:cfg_group):group_q;
 wire parameter_ready=EXTERNAL_HEADER==0 || header_use_ready;
 wire start_video,hit;
 assign start_video=(state==COMPUTE_START || (state==LOAD_START && hit) || (state==LOAD_WAIT && load_done && !load_error)) && video_ready && parameter_ready && !local_cancel && !abort_event;
 assign header_use_valid=EXTERNAL_HEADER!=0 && start_video;
 reg cache_valid,cache_pw,cache_stem;reg [2:0] cache_gi;reg [1:0] cache_group;
 reg [14:0] cache_offset;reg [15:0] cache_image;
 assign hit=cache_valid && cache_pw==mode && cache_stem==stem && cache_gi==gi &&
  cache_group==group_q && cache_offset==weight_offset && cache_image==image;
 // Loader identity is accepted once and remains stable across group changes.
 reg pending_pw,pending_stem;reg [2:0] pending_gi;reg [1:0] pending_group;
 reg [14:0] pending_offset;reg [15:0] pending_image;
 wire load_command=((state==LOAD_START && !hit) || (prefetch && !preload_started)) && !local_cancel && !abort_event;
 wire load_fire=load_command && load_ready;
 // Retire on the actual last acknowledged vector/prefetch completion. Keep
 // DONE only when the receiver applies backpressure; never retire a request.
 // All current writes are acknowledged by video_done. Future weights do
 // not belong to this result; accepted reads drain under load_busy instead.
 assign finish_now=!abort_event && state==COMPUTE_RUN && video_done && !video_error && !chain_group;
 wire wb,wbr,wv_load,wr_load,wl;wire [2:0] wp;wire [5:0] wi;wire [63:0] wd_load;
 fixed_weight_loader loader(.clk(clk),.reset(reset),.cancel(local_cancel),
  .cmd_valid(load_command),.cmd_ready(load_ready),.cmd_pw(load_pw),.cmd_stem(load_stem),
  .cmd_groups(load_groups),.cmd_out_group(load_group),.cmd_weight_offset(load_offset),.cmd_image_tag(image),
  .store_coeff_ready(store_coeff_ready),.store_image_tag(store_image_tag),
  .weight_begin(wb),.weight_begin_ready(wbr),.weight_parts(wp),.weight_valid(wv_load),.weight_ready(wr_load),
  .weight_index(wi),.weight_data(wd_load),.weight_last(wl),
  .done_valid(load_done),.done_ready(1'b1),.busy(load_busy),.error(load_error),.*);
 fixed_video_compute #(.SKIP_BASE_WIDTH(11),.SKIP_OFFSET_PREAPPLIED(1)) engine(.mac_inputs_done(engine_mac_done),.clk(clk),.reset(reset),.cancel(local_cancel),
  .weight_begin(wb),.weight_begin_ready(wbr),.weight_parts(wp),.weight_valid(wv_load),.weight_ready(wr_load),
  .weight_index(wi),.weight_data(wd_load),.weight_last(wl),
  .cmd_valid(start_video),.cmd_ready(video_ready),.cmd_pw(mode),.cmd_stem(stem),.cmd_stride2(stride2),.cmd_nn(nn_view),
  .cmd_stride4(stride4),.cmd_nn4(nn4),
  .cmd_pixels(pixels),.cmd_rows(rows),.cmd_groups(gi),.cmd_group(source_group),
  .cmd_output_x(ox),.cmd_source_width(width),.cmd_source_origin(origin),.cmd_storage_phase(storage_phase),
  .cmd_row_banks(row_banks),.cmd_row_bases(row_bases),
  .cmd_out_groups(go),.cmd_out_group(group_q),.cmd_out_banks(out_banks),.cmd_out_bases(out_bases),.cmd_out_phase(out_phase),
  .cmd_fused(fused),.cmd_main_scaled(main_scaled),.cmd_skip_offset_two(skip_offset_two),.cmd_skip_banks(skip_banks),.cmd_skip_bases(skip_bases),
  .bias(selected_bias),.multiplier(selected_multiplier),.relu_enable(selected_relu),.mask(result_mask),
  .align_multiplier(align_scale),.post_multiplier(post_scale),
  .done_valid(video_done),.done_ready(state==COMPUTE_RUN || state==ABORT),.busy(video_busy),.error(video_error),
  .writes(video_writes),.acks(video_acks),.accepted_vectors(),.results(),.*);
 assign reads=read_count;assign writes=write_base+(state==COMPUTE_RUN ? {19'd0,video_writes}:32'd0);
 assign acks=ack_base+(state==COMPUTE_RUN ? {19'd0,video_acks}:32'd0);
 // These are the actual writer transfers, also used by the unchanged
 // independent tensor scoreboard. No synthetic result/ACK is created here.
 // synthesis translate_off
 wire [5:0] wv=engine.wv,wr=engine.wr;wire [59:0] wa=engine.wa;
 wire [11:0] wh=engine.wh;wire [767:0] wd=engine.wd;
 wire rerror=engine.raw_error,writer_error=engine.writer_error,pool_error=engine.rq_error,pair_error=1'b0;
 // synthesis translate_on
 // synthesis translate_off
 always @(posedge clk)if(!reset && state==COMPUTE_RUN && video_done && !video_error)begin
  if(({1'b0,write_base}+{4'd0,video_writes})>17'd65535 ||
     ({1'b0,ack_base}+{4'd0,video_acks})>17'd65535)
   $fatal(1,"DESCRIPTOR_TOTAL_BOUND");
 end
 // synthesis translate_on
 integer accepted_reads;
 always @*begin accepted_reads=0;for(integer b=0;b<6;b=b+1)accepted_reads=accepted_reads+int'(read_valid[b] && read_ready[b]);end
 always @(posedge clk)begin
  if(reset)begin
   state<=IDLE;loaded<=0;preload_started<=0;preload_done<=0;error<=0;cycles<=0;read_count<=0;write_base<=0;ack_base<=0;
   mode<=0;stem<=0;stride2<=0;nn_view<=0;stride4<=0;nn4<=0;fused<=0;main_scaled<=0;storage_phase<=0;skip_offset_two<=0;
   pixels<=0;rows<=0;gi<=0;go<=0;group_q<=0;source_group<=0;result_mask<=0;ox<=0;width<=0;origin<=0;
   row_banks<=0;row_bases<=0;out_banks<=0;skip_banks<=0;out_bases<=0;skip_bases<=0;out_phase<=0;weight_offset<=0;image<=0;align_scale<=0;post_scale<=0;
   pending_pw<=0;pending_stem<=0;pending_gi<=0;pending_group<=0;pending_offset<=0;pending_image<=0;
   cache_valid<=0;cache_pw<=0;cache_stem<=0;cache_gi<=0;cache_group<=0;cache_offset<=0;cache_image<=0;
  end else begin
   if(state!=IDLE && state!=DONE)cycles<=cycles+32'd1;
   read_count<=read_count+32'(accepted_reads);
   if(pcv && pcr)begin loaded<=!pc_residual;error<=pc_residual;end
   if(local_cancel || load_error || !store_coeff_ready || store_image_tag!=cache_image)cache_valid<=0;
   // A late speculative failure is sticky and visible to the existing outer
   // fault owner. It must never create a second completion for a retired job.
   if(load_done && load_error)error<=1;
   if(load_fire)begin
    pending_pw<=load_pw;pending_stem<=load_stem;pending_gi<=load_groups;
    pending_group<=load_group;pending_offset<=load_offset;pending_image<=image;
    cache_valid<=0;
    if(prefetch)preload_started<=1;
   end
   if(load_done && !load_error && !local_cancel && !abort_event && state!=ABORT)begin
    cache_valid<=1;cache_pw<=pending_pw;cache_stem<=pending_stem;cache_gi<=pending_gi;
    cache_group<=pending_group;cache_offset<=pending_offset;cache_image<=pending_image;
    if(state==COMPUTE_RUN)preload_done<=1;
   end
   if(abort_event)begin state<=ABORT;error<=1;end
   else case(state)
    IDLE:if(cfg_valid && cfg_ready)begin
     state<=LOAD_START;preload_started<=0;preload_done<=0;error<=0;cycles<=0;read_count<=0;write_base<=0;ack_base<=0;
     mode<=cfg_mode_pw;stem<=cfg_stem;stride2<=cfg_stride2;nn_view<=cfg_nn_view;stride4<=cfg_stride4;nn4<=cfg_nn4;fused<=cfg_pw_fuse;main_scaled<=cfg_main_scaled;
     pixels<=cfg_pixels;rows<=cfg_rows;gi<=cfg_in_groups;go<=cfg_out_groups;
     group_q<=cfg_mode_pw || cfg_stem ? 2'd0:cfg_group;source_group<=cfg_group;result_mask<=cfg_mask;
     ox<=cfg_output_x;width<=cfg_source_width;origin<=cfg_source_origin;storage_phase<=cfg_storage_phase;
     row_banks<=cfg_row_banks;row_bases<=cfg_row_bases;out_banks<=cfg_out_banks;out_bases<=cfg_out_bases;out_phase<=cfg_out_phase;
     skip_banks<=cfg_skip_banks;skip_offset_two<=cfg_skip_delta[1];
     // Fold the fixed two-pixel offset into the existing descriptor register.
     // Eleven bits preserve overflow; no new per-pixel counter or latency.
     for(integer r=0;r<4;r=r+1)
      skip_bases[r*11+:11]<={1'b0,cfg_skip_bases[r*10+:10]}+(cfg_skip_delta[1] ? 11'd4:11'd0);
     weight_offset<=cfg_weight_offset;image<=cfg_image_tag;align_scale<=pc_align_multiplier;post_scale<=pc_post_multiplier;
     if(cfg_residual || pc_residual || cfg_row_deltas!=0 || (cfg_skip_delta!=0 && cfg_skip_delta!=2) ||
        (cfg_stem && cfg_out_groups!=1) || (cfg_mask!=8'hff && !(cfg_mode_pw && cfg_mask==7 && cfg_in_groups==1 && cfg_out_groups==1)) ||
        (!(cfg_out_groups==1 || cfg_out_groups==2 || cfg_out_groups==4)) ||
        (!cfg_mode_pw && !cfg_stem && ({1'b0,cfg_group}>=cfg_out_groups || cfg_in_groups!=cfg_out_groups)))begin state<=ABORT;error<=1;end
    end
    HEADER:if(EXTERNAL_HEADER==0 || header_first_ready)state<=LOAD_START;
    LOAD_START:if(hit)begin
     if(start_video)begin state<=COMPUTE_RUN;preload_started<=0;preload_done<=0;end
     else state<=COMPUTE_START;
    end else if(load_ready)state<=LOAD_WAIT;
    LOAD_WAIT:if(load_done)begin
     if(load_error)begin state<=ABORT;error<=1;end
     else if(start_video)begin state<=COMPUTE_RUN;preload_started<=0;preload_done<=0;end
     else state<=COMPUTE_START;
    end
    COMPUTE_START:if(start_video)begin state<=COMPUTE_RUN;preload_started<=0;preload_done<=0;end
    COMPUTE_RUN:if(video_done)begin
     if(video_error)begin state<=ABORT;error<=1;end
     else begin
      write_base<=write_base+{19'd0,video_writes};ack_base<=ack_base+{19'd0,video_acks};
      if(chain_group)begin
       group_q<=group_q+2'd1;
       // A DW descriptor owns every channel group until the last write ACK.
       // The existing prefetch has already requested the next group's weights.
       if(!mode)begin source_group<=source_group+2'd1;weight_offset<=weight_offset+15'd72;end
       // The preceding engine has acknowledged every result. Its next weights
       // were loaded only after its last input; header parameters stayed latched.
       state<=(preload_done || (load_done && !load_error)) ? COMPUTE_START:LOAD_WAIT;
      end else state<=done_ready ? IDLE:DONE;
     end
    end
    ABORT:if(!load_busy && !video_busy && !cancel)state<=DONE;
    DONE:if(done_ready)state<=IDLE;
    default:begin state<=IDLE;error<=1;end
   endcase
  end
 end
endmodule

// Eight independent parameter lanes. 24 RAM10 blocks replace the serial
// three-RAM cache and its 816-bit register payload. RAM outputs own the
// active group until USE group or applied bank changes. FIRST remains a
// lookahead hint; USE must obey use_ready (one synchronous RAM read on miss).
module header_group_cache3(
 input wire clk,reset,cancel,
 input wire write_valid,write_bank,write_scale,input wire [4:0] write_channel,
 input wire [31:0] write_data,input wire write_relu_valid,input wire [31:0] write_relu,
 input wire apply,input wire apply_bank,
 input wire first_valid,input wire [1:0] first_group,output wire first_ready,
 input wire use_valid,input wire [1:0] use_group,output wire use_ready,
 output wire [255:0] bias,multiplier,output wire [7:0] relu,
 output wire busy,output reg error
);
 reg configured,active_bank,have;reg [2:0] held_address;
 reg [31:0] relu_bank[0:1];
 wire flush=reset || cancel;
 wire [2:0] target={active_bank,use_group};
 wire hit=have && held_address==target;
 wire forbidden_write=configured && write_bank==active_bank && (write_valid || write_relu_valid);
 wire re=configured && !flush && !error && !apply && !hit;
 wire scale_bad=write_data[31:18]!={14{write_data[17]}};
 assign first_ready=configured && !flush && !error && !apply;
 assign use_ready=configured && !flush && !error && !apply && hit;
 assign busy=re;
 assign relu=relu_bank[active_bank][use_group*8+:8];
 for(genvar lane=0;lane<8;lane=lane+1)begin:LANE
  wire we=write_valid && write_channel[2:0]==3'(lane) && !flush && !error && !forbidden_write;
  wire [8:0] wa={6'd0,write_bank,write_channel[4:3]};
  wire [8:0] ra={6'd0,target};
  wire [19:0] lo,hi,sc;
  EFX_RAM10 #(.READ_WIDTH(20),.WRITE_WIDTH(20),.OUTPUT_REG(0),.WRITE_MODE("READ_FIRST")) low_bias(
   .WCLK(clk),.WE({2{we && !write_scale}}),.WCLKE(1'b1),.WADDREN(1'b1),.WDATA(write_data[19:0]),.WADDR(wa),
   .RCLK(clk),.RE(re),.RST(1'b0),.RADDREN(1'b1),.RDATA(lo),.RADDR(ra));
  EFX_RAM10 #(.READ_WIDTH(20),.WRITE_WIDTH(20),.OUTPUT_REG(0),.WRITE_MODE("READ_FIRST")) high_bias(
   .WCLK(clk),.WE({2{we && !write_scale}}),.WCLKE(1'b1),.WADDREN(1'b1),.WDATA({8'd0,write_data[31:20]}),.WADDR(wa),
   .RCLK(clk),.RE(re),.RST(1'b0),.RADDREN(1'b1),.RDATA(hi),.RADDR(ra));
  EFX_RAM10 #(.READ_WIDTH(20),.WRITE_WIDTH(20),.OUTPUT_REG(0),.WRITE_MODE("READ_FIRST")) scale(
   .WCLK(clk),.WE({2{we && write_scale}}),.WCLKE(1'b1),.WADDREN(1'b1),.WDATA({1'b0,scale_bad,write_data[17:0]}),.WADDR(wa),
   .RCLK(clk),.RE(re),.RST(1'b0),.RADDREN(1'b1),.RDATA(sc),.RADDR(ra));
  assign bias[lane*32+:32]={hi[11:0],lo};
  assign multiplier[lane*32+:32]={{13{sc[17]}},sc[17]^sc[18],sc[17:0]};
 end
 always @(posedge clk)begin
  if(flush)begin
   configured<=0;active_bank<=0;have<=0;held_address<=0;error<=0;relu_bank[0]<=0;relu_bank[1]<=0;
  end else begin
   if(forbidden_write || (use_valid && !use_ready))error<=1;
   if(write_relu_valid && !forbidden_write)relu_bank[write_bank]<=write_relu;
   if(re)begin have<=1;held_address<=target;end
   if(apply)begin configured<=1;active_bank<=apply_bank;have<=0;end
  end
 end
endmodule
