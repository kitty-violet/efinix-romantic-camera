"""Prepare one causal CSI-to-TMDS board run with the exact mapped RTL.

Warm fixture assumptions: skip ONLY the power-up video delay counter, preload
external DDR payload via Micron's public memory API; actual CPU verifies/loads
weights and starts the real CNN. No forced ready/calibration/commit flags.
"""
import argparse,json,os,re,shutil,subprocess,time
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from prepare_full_board_simulation import prepare
from validate_board_startup import make_tb

def fast_memory(text):
    # Replace only linear lookup bookkeeping. All command, timing, DQS, burst,
    # ODT, refresh and data arrays remain the vendor model's original logic.
    needle='    function get_index;'
    start=text.index(needle);end=text.index('    endfunction',start)+len('    endfunction')
    text=text[:start]+'''    integer native_index [bit [`MAX_BITS-1:0]];
    function get_index;
        input [`MAX_BITS-1:0] addr;
        begin
            // Match the linear model on misses, X/Z addresses and memory reset.
            // Reset clears memory_used by NBA; stale cached addresses must not
            // resurrect old contents or point to a subsequently reused slot.
            get_index = 0;
            memory_index = memory_used;
            if ((^addr) !== 1'bx && native_index.exists(addr)) begin
                if (native_index[addr] < memory_used && address[native_index[addr]] === addr) begin
                    get_index = 1;
                    memory_index = native_index[addr];
                end
            end
        end
    endfunction'''+text[end:]
    needle='                address[memory_used] = addr;'
    assert text.count(needle)==1
    text=text.replace(needle,"                if ((^addr) !== 1'bx) native_index[addr] = memory_used;\n"+needle)
    needle='        memory_used = i;'
    assert text.count(needle)==1
    return text.replace(needle,needle+'''
        native_index.delete();
        for (integer k=0;k<memory_used;k=k+1)
            if ((^address[k]) !== 1'bx) native_index[address[k]]=k;''')

def bench(root,payload,ddr516=False,compute_frame_budget=8,raw_diagnostic=False,warmup_lines=0,camera_before_loader=False,measurement_probes=False):
    # A compute commit precedes its displayed frame. Stopping at exactly three
    # commits can revoke display admission before the third result is shown.
    # Keep a bounded continuous input run; acceptance still requires 3 distinct
    # complete screen frames, and the unchanged firmware retains its watchdog.
    assert 4<=compute_frame_budget<=16
    tb=make_tb(root/'rtl/ti60f225_oob_top.v',ddr516=ddr516)
    tb=tb[:tb.index('integer cycles=0;')]
    # Match the actual receiver's HS_BYTECLK_MHZ=70 setting.
    tb=tb.replace('#2.666666667 i_cam_ck_CLKOUT','#7.142857143 i_cam_ck_CLKOUT')
    camera_names=['i_cam_ck_LP_P_IN','i_cam_ck_LP_N_IN']+[f'cam_d{k}_{s}' for k in range(4) for s in ('HS_IN','LP_P_IN','LP_N_IN')]
    for name in camera_names:
        tb,count=re.subn(r'reg\s+(\[[^]]+\]\s+)?'+name+r'=[01];',lambda m:'wire '+(m.group(1) or '')+name+';',tb)
        assert count==1, f'Camera fixture input declaration not found: {name}'
    for k in range(4):tb=tb.replace(f'cam_d{k}_FIFO_EMPTY=1',f'cam_d{k}_FIFO_EMPTY=0')
    jtag=(HERE/'board_integration/jtag_memory_tasks.svh').read_text()
    names={'jtck':'jtag_inst1_TCK','jsel':'jtag_inst1_SEL','jcapture':'jtag_inst1_CAPTURE','jshift':'jtag_inst1_SHIFT',
           'jupdate':'jtag_inst1_UPDATE','jtdi':'jtag_inst1_TDI','jtdo':'jtag_inst1_TDO','clk':'process_clk_150'}
    for old,new in names.items():jtag=re.sub(r'\b'+old+r'\b',new,jtag)
    meta=json.loads((payload/'payload.json').read_text())
    budget_words=[item for item in meta['mailbox'] if item['address']==0xf9000f18]
    assert len(budget_words)==1
    budget_words[0]['data']=compute_frame_budget
    preloads=[]
    for item in meta['payloads']:
        f=payload/item['file'];assert sha(f)==item['sha256'];data=f.read_bytes();assert len(data)%16==0
        name=f.stem.replace('-','_');dst=root/(name+'.hex')
        dst.write_text(''.join(data[k:k+16][::-1].hex()+'\n' for k in range(0,len(data),16)))
        tb+=f'reg [127:0] preload_{name}[0:{len(data)//16-1}];\n'
        preloads.append(f'$readmemh("{dst.as_posix()}",preload_{name});\n'+
            f'for(integer k=0;k<{len(data)//16};k=k+1)begin\n'+
            f' physical=32\'h{item["physical_address"]:08x}+16*k;\n'+
            f' io_model.memory.memory_write(physical[27:25],physical[24:11],physical[10:1],preload_{name}[k]);\nend\n')
    tb+='''
wire [3:0] camera_hsen;
reg camera_start=0;
wire [15:0] camera_number;wire camera_start_pulse;
integer raw_aw_count=0,raw_w_count=0,raw_b_count=0,ddr_aw_count=0,ddr_w_count=0,ddr_b_count=0;
wire [8:0] video_stage_faults={dut.debayer_marker_error,dut.pair_overflow_error,dut.pair_reservation_error,
 dut.pair_fifo_protocol_error,dut.downscale_protocol_error,dut.upscale_protocol_error,
 dut.output_fifo_protocol_error,dut.hdmi_underflow_error,dut.hdmi_marker_error};
camera_raw10_fixture camera(.clk(core_clk),.byte_clk(i_cam_ck_CLKOUT),.pixel_clk(i_mipi_rx_pclk),
 .reset_n(i_arstn),.start(camera_start),.lp_cp(i_cam_ck_LP_P_IN),.lp_cn(i_cam_ck_LP_N_IN),
 .lp_dp({cam_d3_LP_P_IN,cam_d2_LP_P_IN,cam_d1_LP_P_IN,cam_d0_LP_P_IN}),
 .lp_dn({cam_d3_LP_N_IN,cam_d2_LP_N_IN,cam_d1_LP_N_IN,cam_d0_LP_N_IN}),
 .d0(cam_d0_HS_IN),.d1(cam_d1_HS_IN),.d2(cam_d2_HS_IN),.d3(cam_d3_HS_IN),.hs_enable(camera_hsen),
 .frame_number(camera_number),.frame_start(camera_start_pulse));
reg port_frame_armed=0;
always @(posedge camera_start_pulse)port_frame_armed=1;
always @(posedge camera_hsen[0])if(port_frame_armed)begin
 port_frame_armed=0;$display("CAMERA_PORT_BEGIN camera=%0d t_ns=%0.3f",camera_number,$realtime);$fflush();
end
always @(posedge process_clk_150)begin
 if(dut.raw_frame_bridge.camera_writer.dma_start_valid && dut.raw_frame_bridge.camera_writer.dma_start_ready)begin
  $display("RAW_FRAME_BEGIN frame=%0d camera=%0d t_ns=%0.3f",dut.raw_frame_bridge.camera_writer.token_frame_id,camera_number,$realtime);$fflush();
 end
 if(dut.bridge_write_done)begin $display("RAW_FRAME_STORED frame=%0d t_ns=%0.3f",dut.bridge_write_frame_id,$realtime);$fflush();end
 if(dut.bridge_read_done)begin $display("RAW_FRAME_READ_DONE frame=%0d t_ns=%0.3f",dut.bridge_read_frame_id,$realtime);$fflush();end
 if(dut.pair_pixel_valid && dut.hdmi_source_ready && dut.pair_pixel_sof)begin
  $display("RGB_FRAME_BEGIN frame=%0d t_ns=%0.3f",dut.pair_pixel_frame_id,$realtime);$fflush();
 end
end
'''+jtag+'''
reg [31:0] physical,mail;
reg [31:0] observed_boot_phase=0;
initial begin
 wait(dut.w_arstn===1'b1);repeat(10)@(negedge vid_clk_dvi2);
 // Declared warm video state: bypass only the 903.823ms power-up wait.
 dut.dly_cnt=27'h4000000;
 // The host will use JTAG for mailbox writes anyway. Halt at the real boot
 // waiting point before DDR calibration, avoiding millions of idle CPU polls
 // in the simulator. No internal CPU state is forced; resume remains below.
 wait(observed_boot_phase==32'd1);
 jexpect(32'hf9000f04,32'd1);
 wait(dut.cal_done===1'b1);repeat(1000)@(posedge process_clk_150);
 $display("BOARD_CAL_DONE t_ns=%0.3f",$realtime);$fflush();
'''+''.join(preloads)+'''
 $display("PAYLOAD_PRELOADED bytes=750016 t_ns=%0.3f",$realtime);$fflush();
'''+''.join(f" jwrite(32'h{i['address']:08x},32'h{i['data']:08x});\n" for i in meta['mailbox'])+'''
 jresume();
 wait(dut.control_enable===1'b1 && dut.coeff_ready===1'b1);
 $display("BOARD_CPU_RUN t_ns=%0.3f loaded=%0d",$realtime,dut.loaded_words);$fflush();
 camera_start=1;
end
integer cycles=0;reg [31:0] last_commits=0;real last_commit=0;
wire [31:0] observed_cpu_address=dut.cpu.system_cores_0_logic_cpu_dBus_cmd_payload_address;
wire [31:0] observed_cpu_data=dut.cpu.system_cores_0_logic_cpu_dBus_cmd_payload_data;
wire observed_cpu_store=dut.cpu.system_cores_0_logic_cpu_dBus_cmd_valid &&
 dut.cpu.system_cores_0_dBus_cmd_ready && dut.cpu.system_cores_0_logic_cpu_dBus_cmd_payload_wr;
// Observe the real CPU's accepted bus transfers. No debug halt, memory force,
// synthetic boot flag or replacement processor participates in this monitor.
always @(posedge process_clk_150)if(dut.w_arstn && observed_cpu_store)begin
 if(observed_cpu_address==32'hf9000f04)begin
  observed_boot_phase<=observed_cpu_data;
  $display("BOARD_BOOT_PHASE t_ns=%0.3f value=%h loaded=%0d",$realtime,observed_cpu_data,dut.loaded_words);$fflush();
  if(observed_cpu_data[31])$fatal(1,"Actual CPU loader rejected payload phase=%h",observed_cpu_data);
 end
 if(observed_cpu_address==32'hf9000f20 || observed_cpu_address==32'hf9000f24)
  // Firmware clears these two words at normal boot. Its actual trap handler
  // subsequently publishes failure phase 0x80000005, checked above.
  $display("BOARD_CPU_EXCEPTION_INFO address=%h value=%h",observed_cpu_address,observed_cpu_data);
end
integer rgb_frame_pixels=0;
real source_times[0:15];real begin_times[0:15];integer input_pixels=0;
always @(posedge camera_start_pulse)source_times[camera_number]=$realtime;
always @(posedge process_clk_150)begin
 cycles=cycles+1;
 if(cycles%(camera_start?150000:1500)==0)begin
  $display("BOARD_PROGRESS t_ns=%0.3f enable=%b coeff=%b commits=%0d raw_ready=%b rawdrop=%0d pixels=%0d errors=%b%b%b%b phase=%h loaded=%0d",
   $realtime,dut.control_enable,dut.coeff_ready,dut.committed_frames,dut.bridge_ready_mask,dut.bridge_dropped_frames,input_pixels,
   dut.pipeline_error,dut.camera_write_fault,dut.camera_read_fault,dut.camera_line_fault,observed_boot_phase,dut.loaded_words);$fflush();
 end
 if(dut.pair_pixel_valid && dut.hdmi_source_ready)begin
  input_pixels=input_pixels+1;
  if(dut.pair_pixel_sof)begin
   if(rgb_frame_pixels!=0 && rgb_frame_pixels!=1920*1080)$fatal(1,"Incomplete RGB source frame pixels=%0d",rgb_frame_pixels);
   rgb_frame_pixels=0;
  end
  rgb_frame_pixels=rgb_frame_pixels+1;
  if(rgb_frame_pixels==1920*1080)begin
   $display("RGB_FRAME_COMPLETE frame=%0d t_ns=%0.3f pixels=%0d",dut.pair_pixel_frame_id,$realtime,rgb_frame_pixels);$fflush();
  end
 end
 if(dut.video_processing_pipeline.compute_start)begin
  $display("CNN_BEGIN frame=%0d t_ns=%0.3f",dut.video_processing_pipeline.compute_frame_id,$realtime);$fflush();
 end
 if(dut.committed_frames!=last_commits)begin
  $display("CNN_COMMIT count=%0d frame=%0d t_ns=%0.3f interval_ns=%0.3f cycles=%0d",
   dut.committed_frames,dut.video_processing_pipeline.compute_frame_id,$realtime,$realtime-last_commit,dut.production_cycles);$fflush();
  last_commits=dut.committed_frames;last_commit=$realtime;
 end
 if(cycles>60000000)$fatal(1,"Board frame timeout");
 if(camera_start && (dut.pipeline_error===1'b1 || dut.camera_write_fault===1'b1 || dut.camera_read_fault===1'b1 || dut.camera_line_fault===1'b1 || (|video_stage_faults)===1'b1))begin
  $display("VIDEO_STAGE_FAULT debayer_pair_scalers_output_hdmi=%b",video_stage_faults);
  $display("BOARD_FAULT pipeline=%b write=%b read=%b line=%b boundary=%b wfail=%b wcfg=%b wproto=%b length=%b overflow=%b pack=%b cfifo=%b stream=%b rcfg=%b rproto=%b rfifo=%b manager=%b",
   dut.pipeline_error,dut.camera_write_fault,dut.camera_read_fault,dut.camera_line_fault,dut.camera_boundary_error,
   dut.bridge_write_failed,dut.bridge_write_config_error,dut.bridge_write_protocol_error,dut.bridge_frame_length_error,
   dut.bridge_packer_overflow_error,dut.bridge_packer_protocol_error,dut.bridge_camera_fifo_error,dut.bridge_camera_stream_error,
   dut.bridge_read_config_error,dut.bridge_read_protocol_error,dut.bridge_read_fifo_error,dut.bridge_manager_error);
  $display("RAW_FAULT_STATE writer=%0d dma=%0d words=%0d fifo=%0d beat=%0d AW=%b%b W=%b%b B=%b%b id=%h resp=%b",
   dut.raw_frame_bridge.camera_writer.state,dut.raw_frame_bridge.camera_writer.writer.state,
   dut.raw_frame_bridge.camera_writer.captured_words,dut.raw_frame_bridge.camera_writer.axi_fifo_count,dut.raw_frame_bridge.camera_writer.beat_index,
   dut.raw_axi_awvalid,dut.raw_axi_awready,dut.raw_axi_wvalid,dut.raw_axi_wready,dut.raw_axi_bvalid,dut.raw_axi_bready,dut.raw_axi_bid,dut.raw_axi_bresp);
  $display("FABRIC_FAULT merge=%b fabric=%b raw_aw=%0d raw_w=%0d raw_b=%0d ddr_aw=%0d ddr_w=%0d ddr_b=%0d",
   dut.video_processing_pipeline.pipeline.capture_merge_error,dut.video_processing_pipeline.pipeline.fabric_error,
   raw_aw_count,raw_w_count,raw_b_count,ddr_aw_count,ddr_w_count,ddr_b_count);$fflush();
  $fatal(1,"Actual board path error at %0.3f",$realtime);
 end
end
always @(posedge process_clk_150)if(camera_start)begin
 if(dut.raw_axi_awvalid&&dut.raw_axi_awready)begin
  raw_aw_count=raw_aw_count+1;
  if(raw_aw_count<=4)begin $display("RAW_AW_ACCEPT t_ns=%0.3f address=%h beats=%0d",$realtime,dut.raw_axi_awaddr,dut.raw_axi_awlen+1);$fflush();end
 end
 if(dut.raw_axi_wvalid&&dut.raw_axi_wready)raw_w_count=raw_w_count+1;
 if(dut.raw_axi_bvalid&&dut.raw_axi_bready)raw_b_count=raw_b_count+1;
 if(dut.s_axi_awvalid&&dut.s_axi_awready)ddr_aw_count=ddr_aw_count+1;
 if(dut.s_axi_wvalid&&dut.s_axi_wready)ddr_w_count=ddr_w_count+1;
 if(dut.s_axi_bvalid&&dut.s_axi_bready)ddr_b_count=ddr_b_count+1;
end
// Reject unknown active data in either pure four-state or optional native mode.
always @(posedge process_clk_150)if(dut.w_arstn)begin
 if(dut.m_axi_rvalid && $isunknown({dut.m_axi_rdata,dut.m_axi_rid,dut.m_axi_rresp,dut.m_axi_rlast}))$fatal(1,"DDR active R contains X");
 if(dut.pair_pixel_valid && $isunknown({dut.pair_pixel_rgb,dut.pair_pixel_sof,dut.pair_pixel_eol,dut.pair_pixel_frame_id}))$fatal(1,"RGB active contains X");
end
// Decode the actual board TMDS symbols, including fixed board inversion.
function automatic [7:0] decode(input [9:0] tx);
 reg [9:0] s;reg [7:0] q,d;
 begin s=~tx;q=s[7:0]^{8{s[9]}};d[0]=q[0];
  for(integer k=1;k<8;k=k+1)d[k]=q[k]^q[k-1]^!s[8];decode=d;end
endfunction
reg [23:0] rgb_delay[0:58];reg de_delay[0:58];reg [15:0] frame_delay[0:58];reg [15:0] x_delay[0:58],y_delay[0:58];
reg [23:0] decoded;integer screen_frames=0,screen_pixels=0,screen_frame_pixels=0;real last_screen_begin=0;integer fd;reg [65535:0] seen_screen=0;
initial begin fd=$fopen("tmds_frame_events.csv","w");$fwrite(fd,"event,frame,time_ns,interval_ns\\n");
 for(integer k=0;k<=58;k=k+1)begin de_delay[k]=0;rgb_delay[k]=0;frame_delay[k]=0;x_delay[k]=0;y_delay[k]=0;end
end
always @(posedge hdmi_tx_slow_clk)begin
 for(integer k=58;k>0;k=k-1)begin rgb_delay[k]=rgb_delay[k-1];de_delay[k]=de_delay[k-1];frame_delay[k]=frame_delay[k-1];x_delay[k]=x_delay[k-1];y_delay[k]=y_delay[k-1];end
 rgb_delay[0]={dut.hdmi_tx_rdata,dut.hdmi_tx_gdata,dut.hdmi_tx_bdata};de_delay[0]=dut.hdmi_tx_de && dut.hdmi_source_frame_active;
 frame_delay[0]=dut.hdmi_source_frame_id;x_delay[0]=dut.hdmi_timing_x;y_delay[0]=dut.hdmi_timing_y;
 #1;
 if(de_delay[58])begin
  decoded={decode(tmds_tx_data2_TX_DATA),decode(tmds_tx_data1_TX_DATA),decode(tmds_tx_data0_TX_DATA)};
  if(decoded!==rgb_delay[58])$fatal(1,"TMDS pixel mismatch xy=%0d,%0d got=%h want=%h",x_delay[58],y_delay[58],decoded,rgb_delay[58]);
  screen_pixels=screen_pixels+1;
  if(x_delay[58]==0 && y_delay[58]==0)begin
   screen_frame_pixels=0;
   $display("SCREEN_BEGIN frame=%0d t_ns=%0.3f interval_ns=%0.3f",frame_delay[58],$realtime,$realtime-last_screen_begin);$fflush();
   $fwrite(fd,"begin,%0d,%0.3f,%0.3f\\n",frame_delay[58],$realtime,$realtime-last_screen_begin);last_screen_begin=$realtime;
  end
  screen_frame_pixels=screen_frame_pixels+1;
  if(x_delay[58]==1919 && y_delay[58]==1079)begin
   if(screen_frame_pixels!=1920*1080)$fatal(1,"Incomplete screen scan pixels=%0d",screen_frame_pixels);
   if(!seen_screen[frame_delay[58]])begin screen_frames=screen_frames+1;seen_screen[frame_delay[58]]=1;end
   $display("SCREEN_COMPLETE frame=%0d t_ns=%0.3f pixels=%0d",frame_delay[58],$realtime,screen_pixels);$fflush();
   $fwrite(fd,"end,%0d,%0.3f,0\\n",frame_delay[58],$realtime);$fflush(fd);
   if(screen_frames==3)begin $display("PASS BOARD_THREE_FRAMES_TMDS pixels=%0d",screen_pixels);$fclose(fd);$finish;end
  end
 end
end
endmodule
'''
    # Actual top DDR naming is s_axi_*, not m_axi_*.
    tb=tb.replace('dut.m_axi_','dut.s_axi_')
    if ddr516:
        tb=tb.replace('physical[27:25],physical[24:11],physical[10:1]',
                      'physical[13:11],physical[27:14],physical[10:1]')
        tb=tb.replace('  $fatal(1,"Actual board path error at %0.3f",$realtime);', '''  $display("DDR_NATIVE_STATE busy=%b issued=%b A=%b%b type=%b addr=%h len=%0d size=%0d burst=%b id=%h W=%b%b last=%b wid=%h B=%b%b bid=%h resp=%b R=%b%b",
   dut.u_ddr3_top.address_bridge.write_busy,dut.u_ddr3_top.address_bridge.write_issued,
   dut.u_ddr3_top.avalid,dut.u_ddr3_top.aready,dut.u_ddr3_top.atype,dut.u_ddr3_top.aaddr,dut.u_ddr3_top.alen,dut.u_ddr3_top.asize,dut.u_ddr3_top.aburst,dut.u_ddr3_top.aid,
   dut.u_ddr3_top.wvalid,dut.u_ddr3_top.wready,dut.u_ddr3_top.wlast,dut.u_ddr3_top.wid,
   dut.u_ddr3_top.bvalid,dut.u_ddr3_top.bready,dut.u_ddr3_top.bid,dut.u_ddr3_top.bresp,dut.u_ddr3_top.rvalid,dut.u_ddr3_top.rready);$fflush();
  $fatal(1,"Actual board path error at %0.3f",$realtime);''')
        tb=tb.replace('// Reject unknown active data', '''always @(posedge process_clk_150)if(dut.cal_done)begin
 if(dut.u_ddr3_top.avalid && $isunknown({dut.u_ddr3_top.aid,dut.u_ddr3_top.aaddr,dut.u_ddr3_top.alen,dut.u_ddr3_top.asize,dut.u_ddr3_top.aburst,dut.u_ddr3_top.atype}))$fatal(1,"Native DDR address contains X/Z");
 if(dut.u_ddr3_top.wvalid && $isunknown({dut.u_ddr3_top.wid,dut.u_ddr3_top.wdata,dut.u_ddr3_top.wstrb,dut.u_ddr3_top.wlast}))$fatal(1,"Native DDR write contains X/Z");
 if(dut.u_ddr3_top.bvalid && ($isunknown({dut.u_ddr3_top.bid,dut.u_ddr3_top.bresp}) || dut.u_ddr3_top.bid[7:4]!==0))$fatal(1,"Native DDR B ID/error invalid before narrowing");
 if(dut.u_ddr3_top.rvalid && ($isunknown({dut.u_ddr3_top.rid,dut.u_ddr3_top.rresp}) || dut.u_ddr3_top.rid[7:4]!==0))$fatal(1,"Native DDR R ID/error invalid before narrowing");
end
always @(posedge process_clk_150)if(camera_start)begin
 if(dut.u_ddr3_top.avalid && dut.u_ddr3_top.aready && raw_aw_count<5)begin
  $display("DDR_NATIVE_A t_ns=%0.3f write=%b address=%h len=%0d size=%0d burst=%b id=%h",$realtime,dut.u_ddr3_top.atype,dut.u_ddr3_top.aaddr,dut.u_ddr3_top.alen,dut.u_ddr3_top.asize,dut.u_ddr3_top.aburst,dut.u_ddr3_top.aid);$fflush();end
 if(dut.u_ddr3_top.wvalid && dut.u_ddr3_top.wready && raw_w_count<12)begin
  $display("DDR_NATIVE_W t_ns=%0.3f last=%b strobe=%h data=%h B=%b%b",$realtime,dut.u_ddr3_top.wlast,dut.u_ddr3_top.wstrb,dut.u_ddr3_top.wdata,dut.u_ddr3_top.bvalid,dut.u_ddr3_top.bready);$fflush();end
end
// Reject unknown active data''')
    if camera_before_loader:
        assert not raw_diagnostic and warmup_lines==4
        marker=' $display("BOARD_CAL_DONE t_ns=%0.3f",$realtime);$fflush();'
        assert tb.count(marker)==1
        tb=tb.replace(marker,marker+'''
 // The ideal external camera is already streaming while the real CPU loads.
 // First verify RAW writes with the CPU genuinely halted, then continue this
 // SAME calibrated simulation through boot/CNN/three distinct TMDS frames.
 camera_start=1;#1000000;
 if(raw_w_count<3000)$fatal(1,"Early RAW capture stalled writes=%0d",raw_w_count);
 $display("RAW_CAPTURE_PROGRESS_BEFORE_LOADER writes=%0d bursts=%0d t_ns=%0.3f",raw_w_count,raw_b_count,$realtime);$fflush();
''')
    if raw_diagnostic:
        start=tb.index(' $display("BOARD_CAL_DONE')
        end=tb.index(' camera_start=1;',start)
        tb=tb[:start]+''' $display("RAW_DIAGNOSTIC_CAL_DONE t_ns=%0.3f",$realtime);$fflush();
 // CPU remains halted through real JTAG; this is an isolated RAW/fabric test.
'''+tb[end:]
        warmup_lines=4
        tb=tb.replace('integer cycles=0;reg [31:0] last_commits=0;', '''initial begin wait(camera_start);#1000000;
 if(raw_w_count<3000)$fatal(1,"RAW diagnostic no write progress w=%0d",raw_w_count);
 $display("PASS RAW_DIAGNOSTIC_FIRST_LINES writes=%0d bursts=%0d",raw_w_count,raw_b_count);$finish;end
integer cycles=0;reg [31:0] last_commits=0;''')
    if warmup_lines:
        assert warmup_lines==4
        tb=tb.replace('camera_raw10_fixture camera(', 'camera_raw10_fixture #(.WARMUP_LINES(4)) camera(')
    if measurement_probes:
        assert ddr516
        tb=tb.replace('  last_commits=dut.committed_frames;last_commit=$realtime;',
            '  dump_board_profile("commit",dut.video_processing_pipeline.compute_frame_id);\n  last_commits=dut.committed_frames;last_commit=$realtime;')
        tb=tb.replace('   if(screen_frames==3)begin',
            '   dump_board_profile("screen_complete",frame_delay[58]);\n   if(screen_frames==3)begin')
        assert tb.count('endmodule')==1
        tb=tb.replace('endmodule','`include "board_measurement_probes.svh"\nendmodule')
    if ddr516:
        assert tb.count('endmodule')==1
        tb=tb.replace('endmodule','''initial begin
 wait(dut.cal_done===1'b1);
 $display("BOARD_DDR_RUNTIME_CONFIG WL=%0d BL=%0d AL=%0d TWTR_CK=%0d",
  io_model.memory.write_latency,io_model.memory.burst_length,io_model.memory.additive_latency,io_model.memory.TWTR_TCK);
 if(io_model.memory.write_latency!==5 || io_model.memory.burst_length!==8 ||
    io_model.memory.additive_latency!==0 || io_model.memory.TWTR_TCK!==4)
  $fatal(1,"BOARD_DDR_RUNTIME_CONFIG differs from frozen V4 configuration");
end
ddr516_measurement_guard ddr_guard(
 .user_clk(process_clk_150),.enabled(dut.cal_done),.ck(io_model.ck),
 .reset_n(io_model.rst_n),.cke(io_model.cke),.cs_n(io_model.cs_n),
 .ras_n(io_model.ras_n),.cas_n(io_model.cas_n),.we_n(io_model.we_n),
 .addr(io_model.addr),.bank(io_model.ba),
 .avalid(dut.u_ddr3_top.avalid),.aready(dut.u_ddr3_top.aready),.atype(dut.u_ddr3_top.atype),
 .aaddr(dut.u_ddr3_top.aaddr),.aid(dut.u_ddr3_top.aid),.alen(dut.u_ddr3_top.alen),
 .asize(dut.u_ddr3_top.asize),.aburst(dut.u_ddr3_top.aburst));
endmodule''')
    return tb

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--native',type=Path);p.add_argument('--base',type=Path);p.add_argument('--flat-dsp-proof',type=Path)
 p.add_argument('--ram-word-proof',type=Path,help='Hash-bound Questa four-state proof for public RAM storage acceleration')
 p.add_argument('--simple-ram-word-proof',type=Path,help='Hash-bound Questa proof for equal-width simple dual-port RAM storage')
 p.add_argument('--ring-checker-proof',type=Path,help='Hash-bound Questa four-state proof for equivalent TMDS checker acceleration')
 p.add_argument('--compute-frame-budget',type=int,choices=range(4,17),default=8)
 p.add_argument('--raw-diagnostic',action='store_true')
 p.add_argument('--camera-before-loader',action='store_true',help='Start the external camera after calibration; verify RAW progress and continue the same run through the actual CPU loader and full video')
 p.add_argument('--warmup-lines',type=int,choices=(0,4),default=0,help='Only shorten the explicitly ignored polarity-learning frame; accepted frames remain 1080p30');a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
 base=a.base.resolve() if a.base else ROOT/'tmp/frame-execution/connected-board-boot2';boot=ROOT/'tmp/frame-execution/real-boot-questa3/firmware'
 files,inc,manifest=prepare(root,base,boot)
 if a.flat_dsp_proof:
  proof_root=a.flat_dsp_proof.resolve();proof=json.loads((proof_root/'result.json').read_text())
  evidence=json.loads((proof_root/'inputs.json').read_text())
  assert proof['status']=='DSP24_BOARD_PIPE_HIERARCHY_EQUIVALENCE_PASS'
  assert proof['inputs_sha256']==sha(proof_root/'inputs.json') and proof['log_sha256']==sha(proof_root/'simulation.log')
  primitive=root/'efx_dsp24.v'
  assert sha(primitive)==evidence['original_sha256']
  assert sha(proof_root/'efx_dsp24_flat.v')==proof['flat_sha256']==evidence['flat_sha256']
  shutil.copy2(proof_root/'efx_dsp24_flat.v',primitive)
  record=next(row for row in manifest['inputs'] if Path(row['snapshot'])==primitive)
  record.update(source=str(proof_root/'efx_dsp24_flat.v'),sha256=sha(primitive),
                vendor_original_sha256=evidence['original_sha256'],view='guarded exact-production public DSP simulation specialization; four-state equivalence checked' if proof.get('compact_board') else 'mechanically inlined public pipe hierarchy; four-state equivalence checked')
  shutil.copy2(proof_root/'result.json',root/'dsp_flat_proof.json')
  manifest['inputs'].append(dict(source=str(proof_root/'result.json'),snapshot=str(root/'dsp_flat_proof.json'),sha256=sha(root/'dsp_flat_proof.json')))
  manifest['dsp_pipe_hierarchy_flattened']=True
 if a.ram_word_proof:
  assert not a.native,'Host acceleration evidence is for pure Questa'
  from board_host_speedups import ram_word
  ram_word(root,manifest,a.ram_word_proof)
 if a.simple_ram_word_proof:
  assert not a.native,'Host acceleration evidence is for pure Questa'
  from board_host_speedups import ram_word
  ram_word(root,manifest,a.simple_ram_word_proof,simple=True)
 if a.native:
  files=[f for f in files if f.name!='cnn_video_pipeline.sv'];wrapper=root/'cnn_video_pipeline_dpi.sv';shutil.copy2(a.native/'cnn_video_pipeline_dpi.sv',wrapper);files.append(wrapper)
 package=EF/'ipm/ip/efx_ddr3_soft_controller/testbench'
 extras=[package/'ddr3.v',package/'efx_iddr.v',package/'efx_oddr.v',HERE/'board_integration/ddr_periphery_model.sv',
  ROOT/'tmp/frame-execution/full-board-vendor-probe3/2048Mb_ddr3_parameters.vh',HERE/'board_integration/camera_raw10_fixture.sv',
  EF/'ipm/ip/efx_csi2/testbench/soft_dphy/rx/efx_csi2_tx_modelsim.sv']
 if manifest['ddr516']:
  if not a.native:extras.append(HERE/'board_integration/board_measurement_probes.svh')
  proof_root=ROOT/'tmp/frame-execution/ddr516-v4-pll-startup1'
  proof=json.loads((proof_root/'result.json').read_text())
  assert proof['status']=='PLL_PROTOCOL_CORRECTION_PASS' and proof['v4_clock_plan'] and proof['startup_reset']
  assert sha(proof_root/'EFX_PLL_V3_FAST_PROTOCOL.v')==proof['fast_corrected_sha256']
  shutil.copy2(proof_root/'result.json',root/'pll_startup_proof.json')
  manifest['inputs'].append(dict(source=str(proof_root/'result.json'),snapshot=str(root/'pll_startup_proof.json'),sha256=sha(root/'pll_startup_proof.json')))
  extras=[f for f in extras if f.name!='ddr_periphery_model.sv']
  extras.extend([HERE/'board_integration/ddr516_periphery_model.sv',HERE/'board_integration/ddr516_measurement_guard.sv',EF/'pt/sim_models/verilog/EFX_GPIO_V3.v',
   proof_root/'EFX_PLL_V3_FAST_PROTOCOL.v'])
 for f in extras:
  dst=root/f.name;shutil.copy2(f,dst)
  if f.name=='ddr3.v':dst.write_text(fast_memory(f.read_text()))
  if f.name=='ddr_periphery_model.sv':dst.write_text(f.read_text().replace('.DEBUG(0)','.DEBUG(0),.MEM_BITS(22)'))
  manifest['inputs'].append(dict(source=str(f),snapshot=str(dst),sha256=sha(dst),vendor_original_sha256=sha(f)))
  if f.suffix in ('.v','.sv'):files.append(dst)
 (root/'ddr3_device_ID.vh').write_text('`define sg125\n')
 manifest['inputs'].append(dict(source='Frozen actual device speed-bin selection',snapshot=str(root/'ddr3_device_ID.vh'),sha256=sha(root/'ddr3_device_ID.vh')))
 tb=bench(root,ROOT/'tmp/frame-execution/board-payload2',ddr516=manifest['ddr516'],compute_frame_budget=a.compute_frame_budget,raw_diagnostic=a.raw_diagnostic,warmup_lines=a.warmup_lines,camera_before_loader=a.camera_before_loader,measurement_probes=manifest['ddr516'] and not a.native)
 if a.ring_checker_proof:
  assert not a.native,'Host acceleration evidence is for pure Questa'
  from board_host_speedups import ring_checker
  tb=ring_checker(root,manifest,a.ring_checker_proof,tb)
 (root/'tb_board_startup.sv').write_text(tb);files.append(root/'tb_board_startup.sv')
 # The binary payload was verified during preparation; retain the exact HEX
 # files consumed by readmemh in the frozen manifest as well as the testbench.
 from audit_board_payload_inputs import audit as audit_payload
 payload_audit=audit_payload(root,ROOT/'tmp/frame-execution/board-payload2')
 (root/'payload_integrity_audit.json').write_text(json.dumps(payload_audit,indent=2))
 for row in payload_audit['payloads']:
  manifest['inputs'].append(dict(source=row['binary_source'],snapshot=row['snapshot'],sha256=row['sha256'],binary_sha256=row['binary_sha256']))
 manifest.update(source_files=list(map(str,files)),include_dirs=list(map(str,inc)),native=str(a.native.resolve()) if a.native else None,tb_sha256=sha(files[-1]),
  warm_start='Video delay counter set to steady value; payload initialized through Micron API; real JTAG halts boot waiting loop during DDR calibration, then actual CPU resumes to load/verify coefficients',
  compute_frame_budget=a.compute_frame_budget,required_distinct_screen_frames=3,
  raw_diagnostic=a.raw_diagnostic,
  camera_before_loader=a.camera_before_loader,
  discarded_polarity_learning_frame_lines=4 if a.raw_diagnostic else a.warmup_lines or 1080,
  accepted_camera_frame_geometry=[1920,1080],accepted_camera_period_ns=33333340,
  physical_pass=False,full_baseline=False)
 (root/'inputs.json').write_text(json.dumps(manifest,indent=2))
 sim=Path('E:/quatra/questa_fse/win64');env=os.environ.copy();env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 cmds=[('vlib',[sim/'vlib.exe','work']),('compile',[sim/'vlog.exe','-sv','-timescale','1ns/1ps','+define+den2048Mb+sg125+x16',*[('+incdir+'+str(x))for x in inc],*files])]
 for name,cmd in cmds:
  with (root/(name+'.log')).open('wb') as log:subprocess.run(list(map(str,cmd)),cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 (root/'prepared.json').write_text(json.dumps(dict(status='BOARD_FRAME_COMPILE_PASS',inputs_sha256=sha(root/'inputs.json')),indent=2))
 print('BOARD_FRAME_COMPILE_PASS',flush=True)
if __name__=='__main__':main()
