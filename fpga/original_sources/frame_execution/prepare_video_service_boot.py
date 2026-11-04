"""Derive one reviewed control delta from the frozen, working native boot RTL."""
from run_units import ROOT,HERE,sha
from control_command_repair import once
from pathlib import Path
import json

def generate():
 source=ROOT/'tmp/frame-execution/connected-board-continuous-srl-register1/cnn_preview_boot.sv'
 s=source.read_text()
 s=once(s,'parameter integer BUS_TIMEOUT_CYCLES=4096','parameter integer NO_PROGRESS_CYCLES=300000000,\n parameter integer BUS_TIMEOUT_CYCLES=4096')
 s=once(s,' input wire clk,reset,',' input wire clk,reset,button_event,payload_changed,work_drained,\n output reg prepared,\n output wire draining,\n output reg [31:0] rejected_buttons,')
 s=once(s,'HALT=55;','HALT=55,DRAIN=56,RUN_COMMAND=57,CLEAR_STOP=58;')
 s=once(s,' assign observer_state=state;',
  ' assign observer_state=state;\n assign draining=(state==DRAIN || (state==CLEAR_STOP && stop_command));')
 s=once(s,' reg waiting;',' reg waiting,prepare_only,continuous_mode,button_pending;\n reg payload_invalidated;\n reg [31:0] last_count;\n reg [$clog2(NO_PROGRESS_CYCLES+1)-1:0] drain_ticks;')
 s=once(s,'state==STATUS_RUN || state==RUN_STATUS || state==RUN_COUNT;','state==STATUS_RUN || state==RUN_STATUS || state==RUN_COUNT || state==RUN_COMMAND;')
 s=once(s,'wire run_timeout=running_phase && run_ticks>=RUN_TIMEOUT_CYCLES-1;',
  'wire run_timeout=running_phase && run_ticks >= (continuous_mode ? NO_PROGRESS_CYCLES-1:RUN_TIMEOUT_CYCLES-1);')
 s=once(s,'operations[WAIT_CMD]=op(0,1,0,0);',
  'operations[RUN_COMMAND]=op(0,1,0,0);operations[CLEAR_STOP]=op(1,1,0,0);\n  operations[WAIT_CMD]=op(0,1,0,0);')
 s=once(s,'state<=INIT_CTRL;waiting<=0;',
  'state<=INIT_CTRL;prepared<=0;prepare_only<=0;continuous_mode<=0;button_pending<=0;\n   last_count<=0;drain_ticks<=0;rejected_buttons<=0;payload_invalidated<=0;waiting<=0;')
 s=once(s,'end else if(!terminal || state==WAIT_CMD)begin',
  '''end else if(!terminal || state==WAIT_CMD)begin
   if(button_event)begin
    if(prepared && (state==WAIT_CMD || running_phase))button_pending<=1;
    else if(!(&rejected_buttons))rejected_buttons<=rejected_buttons+1'b1;
   end
   if(payload_changed)prepared<=0;
   if(payload_changed && ((state>=CLEAR_CMD && state<=RUN_COUNT) || running_phase || state==DRAIN || state==CLEAR_STOP))payload_invalidated<=1;''')
 s=once(s,'else if(state==CRC_LO || state==CRC_HI)begin',
  '''else if((payload_invalidated || payload_changed) &&
     ((state>=CLEAR_CMD && state<=RUN_COUNT) || running_phase || state==DRAIN || state==CLEAR_STOP))begin
    prepared<=0;run_allowed<=0;failed<=1;fault<=9;
    // Finish an accepted request before changing its reply interpretation.
    if((!waiting && !cmd_valid) || step)state<=FAIL_STOP;
   end
   else if(state==DRAIN)begin
    if(work_drained)begin drain_ticks<=0;state<=STOP;end
    else if(drain_ticks==NO_PROGRESS_CYCLES-1)reject(10);
    else drain_ticks<=drain_ticks+1'b1;
   end else if(state==CRC_LO || state==CRC_HI)begin''')
 old='''WAIT_CMD:if(rsp_data!=0)begin
     if(rsp_data!=32'h434e4e31)reject(1);
     else begin terminal<=0;state<=CLEAR_CMD;end
    end'''
 new='''WAIT_CMD:begin
     if(rsp_data==32'h4c495645)begin
      if(!prepared || payload_changed)reject(3);
      else begin terminal<=0;continuous_mode<=1;frames<=1023;prepare_only<=0;state<=CLEAR_STOP;end
     end else if(rsp_data!=0)begin
      if(rsp_data!=32'h434e4e31 && rsp_data!=32'h4c4f4144)reject(1);
      else begin terminal<=0;prepared<=0;button_pending<=0;payload_invalidated<=0;
       prepare_only<=rsp_data==32'h4c4f4144;continuous_mode<=0;state<=CLEAR_CMD;end
     end else if(button_pending)begin
      button_pending<=0;
      if(prepared && !payload_changed)begin terminal<=0;continuous_mode<=1;frames<=1023;prepare_only<=0;state<=GET_COUNT;end
     end
    end'''
 s=once(s,old,new)
 s=once(s,'GET_FRAMES:if(rsp_data<1 || rsp_data>512)',
  'GET_FRAMES:if(!prepare_only && (rsp_data<1 || rsp_data>512))')
 s=once(s,'SET_DCRC:state<=GET_COUNT;GET_COUNT:begin base_count<=rsp_data;state<=START;end',
  '''SET_DCRC:begin prepared<=!payload_changed;state<=GET_COUNT;end
    GET_COUNT:begin base_count<=rsp_data;last_count<=rsp_data;run_ticks<=0;
     if(prepare_only)begin prepare_only<=0;state<=STATUS_DONE;end else state<=START;end''')
 s=once(s,'RUN_STATUS:if(|(rsp_data & 32\'d24))reject(6);else state<=RUN_COUNT;',
  '''RUN_STATUS:if(|(rsp_data & 32'd24))reject(6);else state<=continuous_mode ? RUN_COMMAND:RUN_COUNT;
    RUN_COMMAND:if(rsp_data==32'h53544f50)begin run_allowed<=0;button_pending<=0;state<=CLEAR_STOP;end
     else if(rsp_data!=0)reject(1);else state<=RUN_COUNT;
    CLEAR_STOP:if(run_allowed)state<=RUN_COUNT;
     else if(continuous_mode && state==CLEAR_STOP && !terminal && !prepared)reject(3);
     else if(continuous_mode && cmd_data==0)begin
      // WAIT_CMD LIVE and running STOP are distinguished by the dedicated flag.
      state<=prepare_only ? DRAIN:GET_COUNT;
     end''')
 # Use an explicit stop latch; never infer purpose from an unrelated bus value.
 s=once(s,'reg waiting,prepare_only,continuous_mode,button_pending;',
  'reg waiting,prepare_only,continuous_mode,button_pending,stop_command;')
 s=once(s,'prepared<=0;prepare_only<=0;continuous_mode<=0;button_pending<=0;',
  'prepared<=0;prepare_only<=0;continuous_mode<=0;button_pending<=0;stop_command<=0;')
 s=once(s,'frames<=1023;prepare_only<=0;state<=CLEAR_STOP;',
  'frames<=1023;prepare_only<=0;stop_command<=0;state<=CLEAR_STOP;')
 s=once(s,'run_allowed<=0;button_pending<=0;state<=CLEAR_STOP;',
  'run_allowed<=0;button_pending<=0;stop_command<=1;drain_ticks<=0;state<=CLEAR_STOP;')
 start=s.index('    CLEAR_STOP:if(run_allowed)');end=s.index('    RUN_COUNT:',start)
 s=s[:start]+'''    CLEAR_STOP:state<=stop_command ? DRAIN:GET_COUNT;
'''+s[end:]
 s=once(s,"RUN_COUNT:if((rsp_data-base_count)>={22'd0,frames})begin run_allowed<=0;state<=STOP;end else state<=RUN_STATUS;",
  '''RUN_COUNT:begin
     if(continuous_mode && rsp_data!=last_count)run_ticks<=0;
     last_count<=rsp_data;
     if(button_pending || (!continuous_mode && (rsp_data-base_count)>={22'd0,frames}))begin
      button_pending<=0;run_allowed<=0;drain_ticks<=0;state<=DRAIN;
     end else state<=RUN_STATUS;
    end''')
 s=once(s,'// Candidate: bounded continuous preview, 1..512 frames, 60s hard watchdog.',
  '// Video-service control: LOAD prepares once, button/LIVE starts, STOP drains.\n// Legacy CNN1 remains bounded; continuous mode has a progress watchdog.')
 dst=HERE/'variants/video_service/cnn_preview_boot.sv';dst.write_text(s,encoding='utf8')
 # Existing regression exercises full CRC, readback and bus backpressure.
 from validate_continuous_run_control import boot_tb
 t=boot_tb().replace('cnn_boot_sequencer #(', 'cnn_preview_boot #(')
 t=once(t,'wire [5:0] observer_state;',
  'reg button_event=0,payload_changed=0,work_drained=1;wire prepared,draining;wire [31:0] rejected_buttons;\n wire [206:0] native_state;wire [9:0] frame_budget;wire [5:0] observer_state;')
 t=t.replace('module tb_cnn_boot_sequencer;', 'module tb_video_boot_legacy;')
 (dst.parent/'tb_video_boot_legacy.sv').write_text(t,encoding='utf8')
 live=t.replace('module tb_video_boot_legacy;', 'module tb_video_boot_live;')
 live=once(live,'.BUS_TIMEOUT_CYCLES(64)', '.NO_PROGRESS_CYCLES(4096),.BUS_TIMEOUT_CYCLES(64)')
 live=once(live,'reg [31:0] mailbox_command=0;', 'reg freeze_progress=0;reg [31:0] live_commits=0;\n reg [31:0] mailbox_command=0;')
 live=once(live,"if(cmd_data==1)mailbox_command=mode==1 ? 32'hbad0:32'h434e4e31;",
  "if(cmd_data==1)mailbox_command=32'h4c4f4144;")
 line=next(x for x in live.splitlines() if "APB+'h5c:response=" in x)
 live=once(live,line,"    APB+'h5c:response=live_commits;")
 live=once(live,'run_clocks=run_clocks+1;',
  'run_clocks=run_clocks+1;\n    if(!freeze_progress && run_clocks%128==0)live_commits=live_commits+1;')
 first=live.index(' initial begin\n  integer ticks;')
 live=live[:first]+''' task press_key;
 begin @(negedge clk);button_event=1;@(negedge clk);button_event=0;end endtask
 task await_ready;
 integer n;
 begin n=0;while((!terminal || !prepared) && n<600000)begin @(negedge clk);n=n+1;end
  if(!terminal || !prepared || failed || run_allowed || loads!=2560 || diags!=2560)$fatal(1,"prepare-only invalid");end endtask
 task await_run;
 integer n;
 begin n=0;while(!run_allowed && n<2000)begin @(negedge clk);n=n+1;end
  if(!run_allowed || frame_budget!=1023 || !prepared || loads!=2560 || diags!=2560)$fatal(1,"LIVE start or unnecessary reload");end endtask
 initial begin
  integer n;
  expected_crc=32'hffffffff;
  for(integer k=0;k<5120;k=k+1)expected_crc=crc_word(expected_crc,word_at(k));
  expected_crc=~expected_crc;
  repeat(5)@(negedge clk);reset=0;
  press_key();await_ready();
  if(rejected_buttons!=1)$fatal(1,"unprepared key not rejected");
  repeat(50)@(negedge clk);if(run_allowed)$fatal(1,"LOAD started CNN");
  press_key();await_run();
  repeat(24000)@(negedge clk);
  if(!run_allowed || failed || live_commits<100)$fatal(1,"continuous mode used total runtime limit");
  work_drained=0;press_key();
  n=0;while(run_allowed && n<2000)begin @(negedge clk);n=n+1;end
  repeat(200)@(negedge clk);
  if(run_allowed || ctrl!=3 || dut.state!=56 || !draining || terminal)$fatal(1,"STOP cut in-flight work");
  work_drained=1;await_ready();
  if(ctrl!=0 || loads!=2560 || diags!=2560)$fatal(1,"stop not drained");
  mailbox_command=32'h4c495645;await_run();repeat(256)@(negedge clk);
  mailbox_command=32'h53544f50;await_ready();
  if(ctrl!=0 || mailbox_command!=0)$fatal(1,"host STOP not consumed");
  press_key();await_run();
  n=0;while((dut.state!=57 || !dut.waiting) && n<2000)begin @(negedge clk);n=n+1;end
  if(n==2000)$fatal(1,"runtime command read setup not reached");
  @(negedge clk);payload_changed=1;@(negedge clk);payload_changed=0;
  n=0;while(!terminal && n<2000)begin @(negedge clk);n=n+1;end
  if(!failed || fault!=9 || prepared || run_allowed)$fatal(1,"modified payload stayed READY");
  @(negedge clk);reset=1;repeat(4)@(negedge clk);reset=0;
  n=0;while((dut.state!=16 || !dut.waiting) && n<10000)begin @(negedge clk);n=n+1;end
  if(n==10000)$fatal(1,"load read setup not reached");
  payload_changed=1;@(negedge clk);payload_changed=0;
  n=0;while(!terminal && n<2000)begin @(negedge clk);n=n+1;end
  if(!failed || fault!=9 || prepared || run_allowed || ctrl!=0)$fatal(1,"mid-load write not rejected/drained");
  @(negedge clk);reset=1;repeat(4)@(negedge clk);reset=0;await_ready();
  freeze_progress=1;press_key();await_run();
  n=0;while(!terminal && n<10000)begin @(negedge clk);n=n+1;end
  if(!failed || fault!=7 || run_allowed)$fatal(1,"no-progress protection absent");
  $display("PASS VIDEO_BOOT_LIVE load_only=1 key_reuse=1 no_total_timeout=1 stop_drains=1 host_live_stop=1 payload_invalidation=1 no_progress_timeout=1");$finish;
 end
 initial begin #30000000;$fatal(1,"live control test timeout");end
endmodule
'''
 (dst.parent/'tb_video_boot_live.sv').write_text(live,encoding='utf8')
 receipt=dict(source=str(source),source_sha256=sha(source),generated_sha256=sha(dst),
  network_changed=False,hardware_access=False,board_release=False)
 (dst.parent/'boot_derivation.json').write_text(json.dumps(receipt,indent=2));return dst
if __name__=='__main__':print(generate())
