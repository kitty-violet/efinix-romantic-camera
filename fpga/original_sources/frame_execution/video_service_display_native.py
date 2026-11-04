"""Explicit display recovery and observation cut, using existing native buses."""
import re
from native_local_drivers import render_token

P='video_processing_pipeline/'
O='cpu/system_hardJtag_debug_logic_jtagBridge/observation/'
RESET='u_ddr3_top/controller/u_efx_ddr3_soft_controller/genblk1.inst_ddr3_axi/inst_efx_ddr3/top_mc/clk_rstn'
def net(n,b=None):return ('net',n,b)
def explicit_net_assignments(code):
 return re.sub(r'\bwire\s+([A-Za-z_]\w*)\s*=\s*([^;]+);',
  lambda m:'wire '+m[1]+'; assign '+m[1]+'='+m[2]+';',code)
def split_net_declarations(code):
 decl=[]
 def take(m):decl.append(m[0]);return ''
 body=re.sub(r'\bwire\s+(?:\[[^\]]+\]\s*)?[A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*\s*;',take,code)
 return decl,body
def extension(D,removed,edits):
 g,ps,loads=D['g'],D['ps'],D['loads']
 assert net(RESET) in loads  # driven by the original continuous reset alias
 def edit(cell,pin,old,new):
  assert cell not in removed and old in g[cell][1][pin],(cell,pin,old)
  assert old not in edits.get(cell,{}).get(pin,{})
  edits.setdefault(cell,{}).setdefault(pin,{})[old]=new
 # The ready decoder is shared with the scaler. Only these two sinks belong
 # to the reader's pop/ROW_WAIT predicates; keep scaler input logic intact.
 assert ps['LUT__151090']['LUTMASK']==(16,0x0101)
 for cell,pin in [('LUT__151091','I2'),('LUT__151775','I2')]:
  edit(cell,pin,net('n73170_2'),net('service_reader_ready'))
 # All local state has reset polarity checked. Data-only pipeline flops that
 # intentionally have no reset remain unchanged; their valid bits are reset.
 reset_edits=[]
 for cell,(kind,pins) in g.items():
  if kind!='EFX_FF':continue
  local=cell.startswith((P+'output_fifo/',P+'display_reset_pipe')) or cell in (P+'fifo_wr_guard_ready~FF',P+'fifo_rd_guard_ready~FF')
  upscale=cell.startswith(P+'upscaler/')
  old=pins['SR'][0]
  if (local and old==net('w_arstn')) or (upscale and old==net(RESET)):
   assert ps[cell]['SR_POLARITY']==(1,0)
   edit(cell,'SR',old,net('service_video_rst_n' if upscale else 'service_video_arst_n'));reset_edits.append(cell)
 # Insert the pattern at the existing parallel RGB -> HDMI encoder boundary.
 for bit in range(24):
  q=net('hdmi_scheduled_rgb',bit)
  assert q in D['drivers'] and loads[q]
  for cell,pin in loads[q]:edit(cell,pin,q,net('service_output_rgb',bit))
 # Reuse only 183 of 567 retained observation bits. Four separately coherent
 # chunks expose every field without a full-width mux/cross-domain register.
 # The remaining legacy D pins are untouched, including on service phases.
 positions=sorted(int(re.search(r'held_payload\[(\d+)\]',n)[1]) for n in g if n.startswith(O+'held_payload['))
 assert len(positions)==567
 positions=positions[:183]
 mux=[]
 for i,pos in enumerate(positions):
  cell=O+f'held_payload[{pos}]~FF';old=g[cell][1]['D'][0]
  edit(cell,'D',old,net('service_observation_mux',i))
  mux.append(f'assign service_observation_mux[{i}]=service_page_phase<2 ? {render_token(old)} : service_observation_packet[{i}];')
 code=r'''
    wire service_video_inhibit,service_video_discard,service_video_clear,service_video_failed;
    wire [3:0] service_video_fault_code;wire [2:0] service_video_recovery_state;
    wire [31:0] service_video_recoveries;
    wire service_reader_ready=n73170_2 || service_video_discard;
    wire service_video_rst_n=PROCESS_RST_N && !service_video_clear;
    wire service_video_arst_n=w_arstn && !service_video_clear;
    video_display_recovery service_display_recovery(.clk(process_clk_150),.reset(service_reset),
      .writer_fault(WRITE_ERROR),.pixel_fault(READ_ERROR || QUEUE_ERROR),
      .reader_idle(READER_STATE==0),.reader_error(bringup_pipeline_status[3]),
      .inhibit_start(service_video_inhibit),.drain_discard(service_video_discard),.local_reset(service_video_clear),
      .failed(service_video_failed),.fault_code(service_video_fault_code),
      .recovery_count(service_video_recoveries),.state(service_video_recovery_state));
    // A reserved, already decoded mailbox word. No new host/AXI aperture.
    reg service_pattern_request=0,service_clear_toggle=0;
    wire service_diag_write=service_host_write && HOST_ADDRESS==32'hf9000f28 && HOST_DATA[31:16]==16'h5653;
    always @(posedge process_clk_150)begin
      if(service_reset)service_pattern_request<=0;
      else if(service_diag_write)begin
        service_pattern_request<=HOST_DATA[0];
        if(HOST_DATA[8])service_clear_toggle<=!service_clear_toggle;
      end
    end
    (* async_reg="true",syn_srlstyle="registers" *) reg service_clear_meta=0,service_clear_sync=0;
    reg service_clear_seen=0;
    always @(posedge hdmi_tx_slow_clk)begin
      service_clear_meta<=service_clear_toggle;service_clear_sync<=service_clear_meta;service_clear_seen<=service_clear_sync;
    end
    wire service_hdmi_show=SCHED_STATE==1;
    wire service_hdmi_valid=n74771_2;
    wire service_hdmi_marker=service_hdmi_show && hdmi_tx_de && service_hdmi_valid &&
      (((FIFO_X==0 && FIFO_Y==0)!=(hdmi_timing_x==0 && hdmi_timing_y==0)) ||
       ((FIFO_X==1919)!=(hdmi_timing_x==1919)) ||
       ((FIFO_X==1919 && FIFO_Y==1079)!=(hdmi_timing_x==1919 && hdmi_timing_y==1079)) ||
       FIFO_ID!=hdmi_source_frame_id);
    wire service_hdmi_underflow=service_hdmi_show && hdmi_tx_de && !service_hdmi_valid;
    wire service_hdmi_complete=service_hdmi_show && hdmi_tx_de && service_hdmi_valid &&
      !service_hdmi_marker && hdmi_timing_x==1919 && hdmi_timing_y==1079;
    wire [23:0] service_output_rgb;wire [511:0] service_pixel_page,service_health_page,service_path_page;
    wire [127:0] service_pixel_cache;wire [1:0] service_pixel_part;
    wire service_pattern_active,service_pixel_valid,service_pixel_busy,service_pixel_stale;
    wire [31:0] service_pixel_sequence,service_pixel_timeouts;
    video_display_observer service_display_observer(.clk(hdmi_tx_slow_clk),.video_reset(!DISPLAY_RESET[3]),
      .clear_first(service_clear_sync!=service_clear_seen),.pattern_request(service_pattern_request),
      .x({4'd0,hdmi_timing_x[11:0]}),.y({5'd0,hdmi_timing_y[10:0]}),.frame_id(hdmi_source_frame_id),
      .de(hdmi_tx_de),.hs(hdmi_tx_hs),.vs(hdmi_tx_vs),.source_active(service_hdmi_show),.source_valid(service_hdmi_valid),
      .source_completed(service_hdmi_complete),.underflow_event(service_hdmi_underflow),.marker_event(service_hdmi_marker),
      .fifo_write_error(WRITE_ERROR_SYNC),.fifo_read_error(READ_ERROR),.fifo_queue_error(QUEUE_ERROR),
      .fifo_count(FIFO_COUNT),.source_rgb(hdmi_scheduled_rgb),.output_rgb(service_output_rgb),
      .diagnostic_page(service_pixel_page),.pattern_active(service_pattern_active));
    wire service_snapshot_request=OBS_REQUEST;
    reg [31:0] service_process_ticks=0;reg [2:0] service_page_phase=0;reg [7:0] service_page_sequence=0;
    reg [1:0] service_page_part=0;
    video_debug_mailbox #(.WIDTH(128),.SOURCE_WIDTH(512)) service_pixel_snapshot(.dst_clk(process_clk_150),.request(service_snapshot_request),
      .src_clk(hdmi_tx_slow_clk),.source_data(service_pixel_page),.data(service_pixel_cache),
      .request_part(service_page_part),.returned_part(service_pixel_part),
      .valid(service_pixel_valid),.busy(service_pixel_busy),.stale(service_pixel_stale),
      .sequence_id(service_pixel_sequence),.timeout_count(service_pixel_timeouts));
    always @(posedge process_clk_150)begin
      service_process_ticks<=service_process_ticks+1'b1;
      if(service_snapshot_request)begin
        if(service_page_phase==5)begin service_page_phase<=0;service_page_part<=service_page_part+1'b1;end
        else service_page_phase<=service_page_phase+1'b1;
        service_page_sequence<=service_page_sequence+1'b1;
      end
    end
    wire [511:0] service_process_page;
    assign service_process_page={32'h53525631,service_process_ticks,service_video_recoveries,
      {19'd0,service_video_fault_code,service_video_recovery_state,service_video_failed,service_video_clear,
       service_video_discard,service_video_inhibit,service_pattern_request,service_prepared},
      service_capture_admitted,service_capture_skipped,service_capture_orphans,service_compute_admitted,
      committed_frames,service_repeated,dropped_results,service_pixel_sequence,service_pixel_timeouts,
      service_rejected_buttons,{23'd0,FRAME_STATES},
      {21'd0,service_front_bank,service_front_valid,service_draining,service_passing,service_work_drained,
       cal_done_sync,bringup_pipeline_status[3:0]}};
    wire [127:0] service_selected_page;
    assign service_selected_page=service_page_phase==4 ? service_health_page[service_page_part*128+:128] :
      service_page_phase==5 ? service_path_page[service_page_part*128+:128] :
      service_page_phase==3 ? service_process_page[service_page_part*128+:128]:service_pixel_cache;
    wire [1:0] service_selected_part;
    assign service_selected_part=service_page_phase==2 ? service_pixel_part:service_page_part;
    wire [182:0] service_observation_packet;
    assign service_observation_packet={4'd0,service_pixel_stale,service_pixel_busy,service_pixel_valid,
      8'd2,service_page_sequence,27'h2b29800,service_selected_part,service_page_phase,service_selected_page};
    wire [182:0] service_observation_mux;
'''
 repl={'PROCESS_RST_N':RESET,'WRITE_ERROR':P+'output_fifo/write_error','READ_ERROR':P+'output_fifo/read_error',
  'QUEUE_ERROR':P+'output_fifo/qerror','READER_STATE':P+'pipeline/display/state','HOST_ADDRESS':'cpu/host_address',
  'HOST_DATA':'cpu/host_data','SCHED_STATE':P+'scheduler/state','FIFO_X':P+'output_fifo/rx','FIFO_Y':P+'output_fifo/ry',
  'FIFO_ID':P+'fifo_full_id','DISPLAY_RESET':P+'display_reset_pipe','WRITE_ERROR_SYNC':P+'output_fifo/write_error_sync',
  'FIFO_COUNT':P+'output_fifo/raw_count','OBS_REQUEST':O+'n5','FRAME_STATES':P+'frame_states'}
 code=re.sub(r'\b('+'|'.join(repl)+r')\b',lambda m:render_token(net(repl[m[0]])),code)
 code+='\n'+'\n'.join(mux)+'\n'
 known={q[1] for q in D['drivers']}|{q[1] for q in loads}
 refs=set(re.findall(r'\\(\S+)\s',code))
 assert refs<=known,('unknown native reference',sorted(refs-known))
 assert sum(n.startswith(P+'upscaler/') for n in reset_edits)==150
 return explicit_net_assignments(code),dict(observation_positions=positions,reset_cells=reset_edits,
   unchanged_jtag_transport_bits=1024,retained_legacy_phases=[0,1],pixel_phases=[2],process_phases=[3],health_phases=[4],path_phases=[5],
   schema_version=2,chunk_bits=128,chunks_per_page=4,whole_page_atomic=False,unchanged_legacy_payload_bits=384,
   control_address='0xf9000f28',control_magic='0x5653',local_recovery_max=3)
