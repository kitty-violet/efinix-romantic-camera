"""Compile and exercise the exact new native display/observation wiring."""
import argparse,collections,json,re,shutil,time
from pathlib import Path
from run_units import ROOT,HERE,MS,sha,run
from video_service_graph import load
from video_service_display_native import extension,P,O,split_net_declarations
from video_service_modes_native import extension as modes_extension

def main():
 p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 src=a.native.resolve();out=a.root.resolve();assert out.is_relative_to(ROOT/'tmp') and not out.exists();out.mkdir(parents=True)
 proof=json.loads((src/'graft.json').read_text());D=load();edits={};code,receipt=extension(D,set(proof['removed']),edits)
 frozen=(src/'native_source.v').read_text()
 assert code in frozen or split_net_declarations(code)[1] in frozen, 'not the frozen wiring'
 modes,mode_receipt=modes_extension(D,set(proof['removed']),edits)
 assert modes in frozen or split_net_declarations(modes)[1] in frozen,'not the frozen mode wiring'
 code+='\n'+modes
 widths=collections.defaultdict(int)
 for _,pins in D['g'].values():
  for ts in pins.values():
   for kind,n,b in ts:
    if kind=='net':widths[n]=max(widths[n],0 if b is None else b+1)
 names=sorted(set(re.findall(r'\\(\S+)\s',code))-{'w_arstn','process_clk_150','hdmi_tx_slow_clk'})
 assert all(n in widths for n in names), 'test may not invent missing native signals'
 decl='`timescale 1ns/1ps\nmodule display_native_boundary;\nreg process_clk_150=0;always #5 process_clk_150=~process_clk_150;\nreg hdmi_tx_slow_clk=0;always #7 hdmi_tx_slow_clk=~hdmi_tx_slow_clk;\n'
 for n in names:
  w=widths[n];decl+='reg '+(f'[{w-1}:0] ' if w else '')+'\\'+n+' =0;\n'
 decl+='''reg w_arstn=0,service_reset=1,service_host_write=0,n73170_2=0,n74771_2=0;
 reg[31:0] bringup_pipeline_status=0;reg[11:0] hdmi_timing_x=0;reg[10:0] hdmi_timing_y=0;
 reg[15:0] hdmi_source_frame_id=0;reg hdmi_tx_de=0,hdmi_tx_hs=0,hdmi_tx_vs=0;reg[23:0] hdmi_scheduled_rgb=24'h123456;
 reg[31:0] service_capture_admitted=11,service_capture_skipped=22,service_capture_orphans=33,service_compute_admitted=44;
 reg[31:0] committed_frames=55,service_repeated=66,dropped_results=77,service_rejected_buttons=88;
 reg[1:0] service_front_bank=0;reg service_front_valid=0,service_draining=0,service_passing=0,service_work_drained=1,cal_done_sync=1,service_prepared=1;
 reg fixed_boot_run_allowed=0,cap_done=0,cap_failed=0,test_gate_ready=1;
 reg process_pll_locked=1,user_pll_locked=1,pll_locked=1,i_pll_locked=1;
 wire service_source_ready;
'''
 tb=r'''
 task tick;begin @(posedge process_clk_150);#1;end endtask
 task pixel;begin @(posedge hdmi_tx_slow_clk);#1;end endtask
 task snapshot;reg[182:0] packet;reg[2:0] phase;reg[1:0] part;reg[127:0] expected;begin
  phase=service_page_phase;\cpu/system_hardJtag_debug_logic_jtagBridge/observation/n5 =1;
  #1;packet=service_observation_mux;
  if(phase>=2)begin
   part=phase==2 ? service_pixel_part:service_page_part;
   if(packet[159:128] !== (32'h56530000 | (part<<3) | phase) || packet[175:168]!==2 || packet[182:179]!=0)$fatal(1,"native chunk header width/layout");
   case(phase)
    2:expected=service_pixel_cache;
    3:expected=service_process_page[part*128+:128];
    4:expected=service_health_page[part*128+:128];
    5:expected=service_path_page[part*128+:128];
   endcase
   if(packet[127:0]!==expected)$fatal(1,"native chunk selection");
   if(phase==3 && part==1 && packet[127:96]!==55)$fatal(1,"process committed counter layout");
   if(phase==2 && packet[176] && part==3 && (packet[127:96]!==32'h56494431 || packet[31:0]!==1))$fatal(1,"pixel magic/completed counter");
  end
  tick();\cpu/system_hardJtag_debug_logic_jtagBridge/observation/n5 =0;repeat(15)tick();
 end endtask
 initial begin
  repeat(5)tick();w_arstn=1;service_reset=0;
  \u_ddr3_top/controller/u_efx_ddr3_soft_controller/genblk1.inst_ddr3_axi/inst_efx_ddr3/top_mc/clk_rstn =1;
  \video_processing_pipeline/display_reset_pipe =15;
  pixel();
  \video_processing_pipeline/scheduler/state =1;n74771_2=1;hdmi_tx_de=1;
  \video_processing_pipeline/output_fifo/rx =1919;\video_processing_pipeline/output_fifo/ry =1079;
  \video_processing_pipeline/fifo_full_id =7;hdmi_source_frame_id=7;hdmi_timing_x=1919;hdmi_timing_y=1079;
  pixel();hdmi_tx_de=0;hdmi_timing_x=1920;
  repeat(48)snapshot();
  \cpu/host_address =32'hf9000f28;\cpu/host_data =32'h56530001;service_host_write=1;tick();service_host_write=0;
  repeat(4)pixel();if(service_pattern_active)$fatal(1,"pattern switched inside frame");
  hdmi_timing_x=2199;hdmi_timing_y=1124;pixel();hdmi_timing_x=0;hdmi_timing_y=80;pixel();
  if(!service_pattern_active || service_output_rgb!==24'h000000)$fatal(1,"native pattern RGB selection");
  \video_processing_pipeline/pipeline/display/state =2;
  \video_processing_pipeline/output_fifo/write_error =1;tick();
  if(!service_video_discard || !service_reader_ready || service_video_clear)$fatal(1,"native discard bypass");
  repeat(10)tick();if(service_video_clear)$fatal(1,"reset active DDR read");
  \video_processing_pipeline/pipeline/display/state =0;tick();
  if(!service_video_clear || service_video_rst_n || service_video_arst_n)$fatal(1,"local reset not connected");
  \video_processing_pipeline/output_fifo/write_error =0;repeat(140)tick();
  if(service_video_failed || service_video_inhibit || service_video_recoveries!=1)$fatal(1,"native recovery did not rearm");
  \cpu/boot_terminal =1;\cpu/boot_observer_state =7;
  \video_processing_pipeline/frame_states =0;
  \cpu/host_data =32'h56530208;service_host_write=1;tick();service_host_write=0;
  if(service_mode!=2 || !service_bypass_compute)$fatal(1,"native mode configuration");
  service_front_valid=1;service_front_bank=2;
  \cpu/host_data =32'h56530c00;service_host_write=1;tick();service_host_write=0;
  if(!service_inspect_enable || service_virtual_region!=10'h379 || service_inspect_d!=10'h379)$fatal(1,"native locked result read region");
  \video_processing_pipeline/pipeline/disp_araddr =32'h01400000;#1;
  if(service_display_region!=2)$fatal(1,"native input preview read region");
  $display("PASS VIDEO_SERVICE_NATIVE_DISPLAY_BOUNDARY page_layout live_counters pattern drain_reset_rearm");$finish;
 end
 initial begin #100000;$fatal(1,"timeout");end
endmodule
'''
 (out/'display_native_boundary.sv').write_text(decl+code+'\nassign service_gate_ready=test_gate_ready;\n'+tb)
 sources=[out/'display_native_boundary.sv']
 for name in ['video_display_observer.sv','video_debug_mailbox.sv','video_display_recovery.sv','video_service_modes.sv','video_service_health.sv']:
  shutil.copy2(src/name,out/name);sources.append(out/name)
 inputs={str(p):sha(p) for p in [Path(__file__),src/'native_source.v',src/'graft.json',*sources]}
 result=dict(status='RUNNING',hardware_access=False,board_release=False,inputs=inputs);start=time.monotonic()
 try:
  run([MS/'vlib.exe','work'],out,'vlib.log',20)
  run([MS/'vlog.exe','-sv',*[str(p) for p in sources]],out,'compile.log',30)
  log,_=run([MS/'vsim.exe','-c','work.display_native_boundary','-do','run -all; quit -code 0'],out,'simulate.log',30)
  assert 'PASS VIDEO_SERVICE_NATIVE_DISPLAY_BOUNDARY' in log
  assert all(sha(p)==h for p,h in inputs.items())
  result['status']='EXACT_NATIVE_DISPLAY_EXTENSION_BOUNDARY_PASS'
 except Exception as e:result.update(status='FAIL',error=str(e));raise
 finally:
  result['seconds']=time.monotonic()-start;(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='inputs'},indent=2))
if __name__=='__main__':main()
