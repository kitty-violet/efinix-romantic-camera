"""Four-state serial bridge/progress/reset short test; not an APB/board proof."""
import argparse,json,re,os,subprocess,time
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_command_repair import audit_delta
from fixed_jtag_loader import command_scan,decode_response
from read_control_progress import decode
from windows_process_tree import stop_launched_tree

Q=Path('E:/quatra/questa_fse/win64')
def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 board=a.board.resolve();r=a.root.resolve();audit_delta(board)
 assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
 for name in ('EfxSapphireSoc.v','jtag_bringup_snapshot.sv'):(r/name).write_bytes((board/'assembly'/name).read_bytes())
 tb=r'''`timescale 1ns/1ps
module tb;
reg clk=0,tck=0,tdi=0,sel=0,cap=0,shift=0,reset=1;
wire tdo,cv,cl,cb,cr,rv,rr,re,mv,mw;wire [31:0] rd,ma,md;wire [1:0] ms;
reg response=0;reg [31:0] response_data=0;integer commands=0,bits=0;
reg [191:0] first=0;reg [31:0] flags=0,state=0,counts=0,ticks=0;
wire [319:0] payload={ticks,counts,state,flags,first};
JtagBridgeNoTap b(.diagnostic_payload(payload),.diagnostic_levels(8'h3f),
 .io_ctrl_tdi(tdi),.io_ctrl_enable(sel),.io_ctrl_capture(cap),.io_ctrl_shift(shift),
 .io_ctrl_update(1'b0),.io_ctrl_reset(1'b0),.io_ctrl_tdo(tdo),
 .io_remote_cmd_valid(cv),.io_remote_cmd_ready(cr),.io_remote_cmd_payload_last(cl),.io_remote_cmd_payload_fragment(cb),
 .io_remote_rsp_valid(rv),.io_remote_rsp_ready(rr),.io_remote_rsp_payload_error(re),.io_remote_rsp_payload_data(rd),
 .io_systemClk(clk),.debugCd_logic_outputReset(reset),.jtagCtrl_tck(tck));
SystemDebugger d(.io_remote_cmd_valid(cv),.io_remote_cmd_ready(cr),.io_remote_cmd_payload_last(cl),.io_remote_cmd_payload_fragment(cb),
 .io_remote_rsp_valid(rv),.io_remote_rsp_ready(rr),.io_remote_rsp_payload_error(re),.io_remote_rsp_payload_data(rd),
 .io_mem_cmd_valid(mv),.io_mem_cmd_ready(1'b1),.io_mem_cmd_payload_address(ma),.io_mem_cmd_payload_data(md),
 .io_mem_cmd_payload_wr(mw),.io_mem_cmd_payload_size(ms),.io_mem_rsp_valid(response),.io_mem_rsp_payload(response_data),
 .io_systemClk(clk),.debugCd_logic_outputReset(reset));
always #3.333 clk=~clk;
always @(posedge clk)begin
 response<=!reset&&mv;
 if(!reset&&cv)bits=bits+1;
 if(!reset&&mv)begin
  commands=commands+1;
  if(ma!==32'hf8100000||mw!==0||ms!==2)$fatal(1,"decoded command wrong %h %b %b",ma,mw,ms);
  response_data<=32'h54493630;
 end
end
task tick;begin #500;tck=1;#500;tck=0;end endtask
task scan(input [385:0] packet,input integer n,output [385:0] rx);
begin sel=1;cap=1;shift=0;tick;cap=0;shift=1;rx=0;
 for(integer i=0;i<n;i=i+1)begin tdi=packet[i];#1;rx[i]=tdo;tick;end
 shift=0;tick;sel=0;tick;end endtask
reg [385:0] raw;integer before_commands,before_bits;
task snapshot;
begin before_commands=commands;before_bits=bits;scan(2,386,raw);scan(3,386,raw);
 if(raw[33:2]!==32'h4f465032||raw[49:42]!==2||raw[34]!==1||raw[35]!==0)$fatal(1,"bad header %h",raw);
 if(raw[385:66]!==payload)$fatal(1,"incoherent or truncated payload %h %h",raw[385:66],payload);
 if(commands!=before_commands||bits!=before_bits)$fatal(1,"snapshot injected command");
 $display("PROGRESSRAW %h",raw);end endtask
task identity;
begin scan(COMMAND,77,raw);scan(1,36,raw);
 if(raw[35:2]!=={32'h54493630,2'b01})$fatal(1,"response wrong %h",raw);
 $display("BUSRAW %h",raw);end endtask
initial begin
 repeat(12)tick;reset=0;#1000;
 first=192'h12345678_00000000_00000000_f8100000_c0700400_00000001;
 flags=32'h00002200;state={3'd0,13'd55,6'd7,10'h002};counts=32'h12345678;ticks=32'h89abcdef;
 identity;snapshot;
 // Reset while still shifting: terminating a short scan first would emit LAST
 // and submit a malformed complete packet, not model an interrupted reset.
 sel=1;cap=1;shift=0;tick;cap=0;shift=1;tdi=0;repeat(18)tick;
 reset=1;shift=0;tick;sel=0;repeat(12)tick;reset=0;repeat(12)tick;
 identity;
 first=0;flags=0;state=1;counts=32'hffffffff;ticks=0;snapshot;
 if(commands!=2)$fatal(1,"unexpected command count %0d",commands);
 $display("PASS COMMAND_CDC_LIVE_FIRSTFAULT_RESET commands=%0d",commands);$finish;
end
initial begin #3000000;$fatal(1,"timeout");end
endmodule
'''.replace('COMMAND',"386'h%x"%command_scan(False,0xf8100000)[0],1)
 (r/'tb.sv').write_text(tb)
 inputs={f.name:sha(f) for f in r.iterdir() if f.is_file()};(r/'inputs.json').write_text(json.dumps(dict(files=inputs,board=str(board),board_inputs_sha256=sha(board/'inputs.json')),indent=2))
 env=os.environ.copy();env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 start=time.monotonic()
 def run(args,name,budget):
  with (r/name).open('x') as log:
   proc=subprocess.Popen([str(Q/args[0]),*args[1:]],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT)
   try:code=proc.wait(timeout=budget)
   except subprocess.TimeoutExpired:stop_launched_tree(proc);raise
  text=(r/name).read_text(errors='replace');assert code==0 and not re.search(r'(?m)^#?\s*\*\* (Error|Fatal)',text),(name,text[-2500:]);return text
 run(['vlib.exe','work'],'vlib.log',15)
 run(['vlog.exe','-sv','EfxSapphireSoc.v','jtag_bringup_snapshot.sv','tb.sv'],'compile.log',25)
 log=run(['vsim.exe','-c','work.tb','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],'simulation.log',50)
 assert 'PASS COMMAND_CDC_LIVE_FIRSTFAULT_RESET' in log
 records=[decode(int(h,16)) for h in re.findall('PROGRESSRAW ([0-9a-fA-F]+)',log)]
 assert len(records)==2 and records[0]['live']['ticks']==0x89abcdef and records[0]['live']['bus_age']==55
 assert records[0]['live']['received_bits']==0x78 and records[0]['live']['replies']==0x12
 assert records[1]['live']['control_state']['raw']=='001' and records[1]['live']['accepted_commands']==255
 assert [decode_response(int(h,16)) for h in re.findall('BUSRAW ([0-9a-fA-F]+)',log)]==[0x54493630]*2
 for name,h in inputs.items():assert sha(r/name)==h
 result=dict(status='COMMAND_PROGRESS_RTL_SERIAL_RESET_PASS',records=records,seconds=time.monotonic()-start,inputs_sha256=sha(r/'inputs.json'),log_sha256=sha(r/'simulation.log'),scope='Actual serial bridge and debugger; synthetic response and observation payload, not mapped APB or board proof',hardware_access=False,board_release=False)
 (r/'result.json').write_text(json.dumps(result,indent=2));print(result['status'],result['seconds'])
if __name__=='__main__':main()
