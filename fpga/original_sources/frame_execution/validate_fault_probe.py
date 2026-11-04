"""Bounded real adapter regressions and clock/reset/serial observation tests."""
import argparse,json,os,re,shutil,subprocess,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,sha
from fault_probe_board import BASE,audit_delta
from windows_process_tree import stop_launched_tree
Q=Path('E:/quatra/questa_fse/win64')
CLOCK_TB=r'''`timescale 1ns/1ps
module tb_clock;
reg clk=0,core=0,tdqss=0,tac=0,twd=0,core_run=1,sysn=1,ddrn=0,cke=0;
always #3.333 clk=~clk;
always #2.5 if(core_run)core=~core;
always #1.25 tdqss=~tdqss;
always #1.25 tac=~tac;
initial begin #0.625;forever #1.25 twd=~twd;end
wire[191:0] p;
bringup_clock_probe dut(clk,core,tdqss,tac,twd,sysn,ddrn,cke,p);
reg[23:0] c0,c1;reg[191:0] before_reset;
initial begin
 #512100;
 if(p[23:0]<399 || p[23:0]>401 || p[47:24]<799 || p[47:24]>801 ||
    p[71:48]!=p[47:24] || p[95:72]+1<p[47:24] || p[95:72]>p[47:24]+1 ||
    p[119:96]<299 || p[119:96]>301)$fatal(1,"clock ratio %h",p);
 @(negedge core);core_run=0;#100;c0=p[23:0];c1=p[47:24];
 #20000;if(p[23:0]!==c0 || p[47:24]<=c1)$fatal(1,"stopped clock");core_run=1;
 ddrn=1;cke=1;#30000;ddrn=0;cke=0;#40000;ddrn=1;cke=1;#100;
 if(p[191:176]!==16'h0102 || p[159:144]<22 || p[159:144]>25 ||
    p[175:160]<16 || p[175:160]>19)$fatal(1,"DDR pulse events %h",p);
 before_reset=p;sysn=0;#100;sysn=1;#100;
 if(p[191:176]!==before_reset[191:176] || p[23:0]<before_reset[23:0])$fatal(1,"history lost on reset");
 $display("PASS CLOCK_PROBE four_clocks=4 stopped_clock=1 ddr_edges=3 reset_retention=1");$finish;
end
initial begin #800000;$fatal(1,"clock probe timeout");end
endmodule
'''
SERIAL_TB=r'''`timescale 1ns/1ps
module tb_serial;
reg clk=0,tck=0,request=0,poll=0,shift=0;reg[703:0] payload=0;reg[7:0] levels=0;
wire tdo;
always #3.333 clk=~clk;
jtag_bringup_snapshot dut(clk,tck,request,poll,shift,payload,levels,tdo);
task tick;begin #50;tck=1;#50;tck=0;end endtask
reg[767:0] raw;reg[703:0] held;
task snap;
 begin
  request=1;tick;request=0;repeat(8)tick;
  held=payload;poll=1;tick;poll=0;shift=1;raw=0;
  for(integer i=0;i<768;i=i+1)begin #1;raw[i]=tdo;tick;end
  shift=0;
  if(raw[31:0]!==32'h4f465034 || raw[47:40]!==4 || raw[33:32]!==1 || raw[767:64]!==held)
   $fatal(1,"OFP4 snapshot mismatch %h",raw);
  $display("OFP4RAW %h",raw);
 end
endtask
initial begin
 #100;levels=8'ha5;
 for(integer k=0;k<22;k=k+1)payload[32*k+:32]=32'h913457a0+k;
 snap;
 payload=~payload;levels=8'h5a;snap;
 $display("PASS OFP4_SERIAL bits=768 patterns=2");$finish;
end
initial begin #250000;$fatal(1,"serial probe timeout");end
endmodule
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--board',required=True,type=Path);p.add_argument('--root',required=True,type=Path);a=p.parse_args()
 board=a.board.resolve();root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
 audit_delta(board);root.mkdir()
 for n in ('sapphire_axi32_to_axi128.sv','bringup_clock_probe.sv','jtag_bringup_snapshot.sv'):shutil.copy2(board/'assembly'/n,root/n)
 tb=(HERE.parent/'ddr_fabric/tb_sapphire_axi32_to_axi128.sv').read_text()
 tb=tb.replace('    ) dut(.*);','    ) dut(.*);\n    wire [127:0] adapter_diagnostic;')
 assert 'wire [127:0] adapter_diagnostic;' in tb
 monitor=r'''
reg [31:0] previous_live=0,current_before_edge=0;
reg [95:0] held_first=0;
reg previous_valid=0;
integer probe_captures=0;
always @(posedge clk)begin
 if(!rst_n)begin previous_live=0;previous_valid=0;held_first=0;end
 else begin
  current_before_edge=adapter_diagnostic[127:96];
  if(!adapter_diagnostic[95] && protocol_error)begin
   #1;
   if(!adapter_diagnostic[95] || adapter_diagnostic[31:0]!==previous_live)
    $fatal(1,"adapter pre-error capture incorrect");
   probe_captures=probe_captures+1;
  end else #1;
  if(previous_valid && adapter_diagnostic[95:0]!==held_first)$fatal(1,"first fault overwritten");
  previous_valid=adapter_diagnostic[95];held_first=adapter_diagnostic[95:0];
  previous_live=current_before_edge;
 end
end
'''
 tb=tb.replace('endmodule',monitor+'\nendmodule')
 (root/'tb_adapter.sv').write_text(tb);(root/'tb_clock.sv').write_text(CLOCK_TB);(root/'tb_serial.sv').write_text(SERIAL_TB)
 inputs=dict(board=str(board),board_inputs_sha256=sha(board/'inputs.json'),runner_sha256=sha(__file__),
  files={f.name:sha(f) for f in root.iterdir()},hardware_access=False)
 (root/'inputs.json').write_text(json.dumps(inputs,indent=2))
 env=os.environ.copy();env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 def run(args,name,budget):
  start=time.monotonic()
  with (root/name).open('x') as f:
   proc=subprocess.Popen([str(Q/args[0]),*args[1:]],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
   try:code=proc.wait(timeout=budget)
   except subprocess.TimeoutExpired:stop_launched_tree(proc);raise
  body=(root/name).read_text(errors='replace');assert code==0 and not re.search(r'(?m)^#?\s*\*\* (Error|Fatal)',body),(name,body[-2000:])
  return dict(log=name,seconds=time.monotonic()-start,sha256=sha(root/name),passes=re.findall(r'^# (PASS .*)$',body,re.M))
 try:
  run(['vlib.exe','work'],'vlib.log',15)
  run(['vlog.exe','-sv',*[f.name for f in root.glob('*.sv')]],'compile.log',40)
  jobs=[('clock','tb_clock',[]),('serial','tb_serial',[])]
  for seed in (71,1027):
   for tr,bad in ((0,0),(1,0),(1,1)):
    jobs.append((f'adapter-{seed}-{tr}-{bad}','tb_sapphire_axi32_to_axi128',[f'-gTRANSLATION={tr}',f'-gBAD_MAPPING={bad}',f'+SEED={seed}']))
  results=[]
  for name,top,opts in jobs:
   row=run(['vsim.exe','-c','work.'+top,*opts,'-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],name+'.log',60)
   assert row['passes'],row;results.append(row)
  for n,h in inputs['files'].items():assert sha(root/n)==h
  result=dict(status='FAULT_PROBE_CLOCK_RESET_SERIAL_AND_REAL_ADAPTER_PASS',results=results,inputs_sha256=sha(root/'inputs.json'),hardware_access=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
