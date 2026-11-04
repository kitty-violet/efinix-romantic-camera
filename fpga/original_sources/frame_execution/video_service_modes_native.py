"""Mode cuts: writer input, display region selection and read-only CPU window."""
from video_service_display_native import net,explicit_net_assignments
from native_local_drivers import render_token
P='video_processing_pipeline/';R=P+'pipeline/';F=R+'fabric/'
def extension(D,removed,edits):
 g,loads=D['g'],D['loads'];extra=[]
 def edit(cell,pin,old,new):
  assert cell not in removed and old in g[cell][1][pin],(cell,pin,old)
  assert old not in edits.get(cell,{}).get(pin,{})
  edits.setdefault(cell,{}).setdefault(pin,{})[old]=new
 for bit in range(24):
  q=net(P+'down_rgb',bit)
  for cell,pin in loads[q]:
   assert cell.startswith('LUT__243'),cell
   edit(cell,pin,q,net('service_selected_rgb',bit))
 for bit in range(16):
  q=net(P+'down_frame_id',bit)
  chosen=[(n,p) for n,p in loads[q] if n.startswith(('camera_burst_capture/','LUT__243'))]
  assert len(chosen)==2,(bit,chosen)
  for n,p in chosen:edit(n,p,q,net('service_selected_id',bit))
 for marker in ('sof','eol','eof'):
  q=net(P+'downscaler/out_'+marker)
  for n,p in loads[q]:
   if n.startswith('LUT__243'):edit(n,p,q,net('service_selected_'+marker))
 for n,p in [('LUT__243215','I1'),('LUT__243429','I0')]:
  edit(n,p,net('raw_capture_abort'),net('service_capture_abort'))
 for bit,n in [(22,'LUT__159599'),(23,'LUT__159603'),(24,'LUT__159607')]:
  edit(n,'I0',net(R+'disp_araddr',bit),net('service_display_region',bit-22))
 # raddr is the accepted CPU READ address. Preserve the original validation,
 # low address bits/lane, state machine, and every write-address connection.
 # Actual native translation is raddr - E1000000 + 04000000 (the top-level
 # override, not the stale module default 05000000). A locked frame
 # therefore uses virtual DD000000 + slot-base + offset. Ten upper bits are
 # fixed so a large/malicious offset cannot escape the selected 4 MiB slot.
 for bit in range(22,32):
  n=F+(f'cpu128_araddr[{bit}]~FF' if bit<24 else f'cpu_width_adapter/raddr[{bit}]~FF')
  old=g[n][1]['D'][0];new=net('service_inspect_d',bit-22)
  edit(n,'D',old,new)
  extra.append(f'assign service_inspect_d[{bit-22}]=service_inspect_enable ? service_virtual_region[{bit-22}]:{render_token(old)};')
 code=r'''
 wire [1:0] service_mode,service_inspect_bank;wire service_inspect_enable,service_inspect_output,service_inspect_changed;
 wire service_bypass_compute,service_allow_capture,service_auto_stop,service_gate_ready,service_selected_valid;
 wire [23:0] service_selected_rgb;wire[15:0] service_selected_id;
 wire service_selected_sof,service_selected_eol,service_selected_eof;
 wire [31:0] service_mode_rejected;
 wire service_capture_abort=raw_capture_abort && service_mode!=2;
 wire service_terminal_idle=\cpu/boot_terminal  && \cpu/boot_observer_state ==7 && !fixed_boot_run_allowed;
 video_service_modes service_modes(.clk(process_clk_150),.reset(service_reset),
  .config_write(service_diag_write),.config_data(\cpu/host_data ),
  .config_idle(service_terminal_idle && service_work_drained && \video_processing_pipeline/frame_states ==0 && \video_processing_pipeline/pipeline/display/state ==0),
  .stopped_drained(service_terminal_idle && service_work_drained),.front_valid(service_front_valid),.front_bank(service_front_bank),
  .clear_window(service_host_write && \cpu/host_address ==32'hf9000f00 && \cpu/host_data ==32'h4c4f4144),
  .run_allowed(fixed_boot_run_allowed),.capture_done(cap_done),.capture_failed(cap_failed),
  .result_done(service_bypass_compute ? cap_done:(\video_processing_pipeline/pipeline/compute_active  && \video_processing_pipeline/pipeline/prod_frame_valid )),
  .source_valid(\video_processing_pipeline/down_valid ),.source_ready(service_source_ready),.gate_ready(service_gate_ready),
  .source_rgb(\video_processing_pipeline/down_rgb ),.source_id(\video_processing_pipeline/down_frame_id ),
  .source_sof(\video_processing_pipeline/downscaler/out_sof ),.source_eol(\video_processing_pipeline/downscaler/out_eol ),.source_eof(\video_processing_pipeline/downscaler/out_eof ),
  .selected_valid(service_selected_valid),.selected_rgb(service_selected_rgb),.selected_id(service_selected_id),
  .selected_sof(service_selected_sof),.selected_eol(service_selected_eol),.selected_eof(service_selected_eof),
  .bypass_compute(service_bypass_compute),.allow_capture(service_allow_capture),.auto_stop(service_auto_stop),.inspect_changed(service_inspect_changed),
  .mode(service_mode),.inspect_enable(service_inspect_enable),.inspect_output(service_inspect_output),.inspect_bank(service_inspect_bank),.rejected(service_mode_rejected));
 wire [2:0] service_original_display_region;
 assign service_original_display_region=\video_processing_pipeline/pipeline/disp_araddr [24:22];
 wire[2:0] service_display_region;
 assign service_display_region=service_bypass_compute && service_original_display_region>=3 && service_original_display_region<=5 ?
  service_original_display_region-3'd3:service_original_display_region;
 wire[2:0] service_inspect_region;
 assign service_inspect_region={1'b0,service_inspect_bank}+(service_inspect_output ? 3'd3:3'd0);
 wire[9:0] service_virtual_region;
 assign service_virtual_region=10'h374+{7'd0,service_inspect_region};
 wire[9:0] service_inspect_d;
'''.replace('\\','\\')
 # Raw string uses one escaped identifier marker in the emitted Verilog.
 code=code.replace('\\\\','\\')
 code+='\n'+health_extension(D)
 return explicit_net_assignments(code+'\n'+'\n'.join(extra)),dict(
  mode_codes={'LIVE_A':0,'INPUT_PREVIEW':1,'DDR_PATTERN':2,'FROZEN_A':3},
  mode_change_requires='terminal WAIT_CMD, all slots empty, capture/CNN/reader idle; SW1 between occupied runs',
  read_window='existing e1 read aperture, locked stopped front, aliases confined to one 4 MiB input/result slot',
  write_addresses_unchanged=True,cpu_read_region_ffs=10)

def health_extension(D):
 known={q[1] for q in D['loads']}|{q[1] for q in D['drivers']}
 def n(name):assert name in known,name;return render_token(net(name))
 def w(name):return "32'("+n(name)+')'
 def pair(a,b):return "{16'("+n(a)+"),16'("+n(b)+")}"
 reader=R+'display/';up=P+'upscaler/';arb=F+'shared_arbiter/'
 capxy=pair('camera_burst_capture/y','camera_burst_capture/x')
 readxy=pair(reader+'pixel_y',reader+'pixel_x')
 outxy=pair(up+'output_y',up+'output_x')
 flags=' | '.join(w(name)+f'<<{shift}' for name,shift in [(P+'frame_states',0),('cap_state',9),(reader+'state',12),(up+'state',16),('raw_capture_abort',20),('raw_frontend/ack',21)])
 context='{'+','.join([capxy,readxy,outxy,flags])+'}'
 fields=["32'h50415431",
  "{26'd0,service_inspect_bank,service_inspect_output,service_inspect_enable,service_mode}",
  w('camera_burst_capture/frame_id'),w(P+'down_frame_id'),w(R+'disp_araddr'),
  w(reader+'line_offset'),capxy,readxy,outxy,
  w(arb+'read_beats_remaining')+' | '+w(arb+'read_owner')+'<<16 | '+w(arb+'read_active')+'<<18 | '+w(arb+'read_pending')+'<<19',
  w(arb+'write_beats_remaining')+' | '+w(arb+'write_owner')+'<<16 | '+w(arb+'write_active')+'<<18 | '+w(arb+'write_pending')+'<<19 | '+w(arb+'write_data_complete')+'<<20',
  flags,'service_mode_rejected',"{8'd0,service_selected_rgb}","{16'd2200,16'd1125}",'service_maximum_gap']
 assert len(fields)==16
 return f'''
 wire[31:0] service_maximum_gap;
 wire[127:0] service_live_context;
 assign service_live_context={context};
 assign service_path_page={{{', '.join(fields)}}};
 video_service_health service_health(.clk(process_clk_150),.reset(service_reset),
  .clear_first(service_diag_write && {n('cpu/host_data')} [8]),.run_allowed(fixed_boot_run_allowed),
  .source_sof({n(P+'down_valid')} && service_source_ready && {n(P+'downscaler/out_sof')}),
  .calibrated(cal_done_sync),.locks({{process_pll_locked,user_pll_locked,pll_locked,i_pll_locked}}),
  .faults({{3'd0,{n('cpu/boot_failed')},bringup_pipeline_status[3],{n(P+'output_fifo/write_error')},bringup_pipeline_status[0],raw_capture_abort}}),
  .ticks(service_process_ticks),.commits(committed_frames),.live_context(service_live_context),
  .read_busy({n(arb+'read_active')} || {n(arb+'read_pending')}),.write_busy({n(arb+'write_active')} || {n(arb+'write_pending')}),
  .read_remaining({n(arb+'read_beats_remaining')}),.write_remaining({n(arb+'write_beats_remaining')}),
  .maximum_service_gap(service_maximum_gap),.page(service_health_page));
'''
