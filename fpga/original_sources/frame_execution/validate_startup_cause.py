"""Bounded regressions, intentional fault capture, and raw primitive sampling."""
import argparse,json,os,re,shutil,subprocess,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from startup_cause_board import audit_delta
from read_startup_cause import cell_expected,cell_decode,decode
from validate_fault_probe import CLOCK_TB,SERIAL_TB
from windows_process_tree import stop_launched_tree
Q=Path('E:/quatra/questa_fse/win64')
CELL_TB=r'''`timescale 1ns/1ps
module tb_cells;
parameter integer FAULT=0;
reg clk=0,ddr=0,rst=0,run_ddr=1;
always #3.333 clk=~clk;
always #1.25 if(run_ddr)ddr=~ddr;
wire [127:0] p;
bringup_cell_pair_probe dut(clk,ddr,rst,p);
reg[127:0] held;
// A short combinational decode glitch after a binary-index change must not
// reach the asynchronous SR pins. It ends well before the next clock edge.
initial if(FAULT==5)begin
 wait(dut.local_cells.index==2);#0.2;
 force dut.local_cells.reset_decoded=0;#0.2;release dut.local_cells.reset_decoded;
end
initial begin
 if(FAULT==1)force dut.local_cells.pair_under_test.P=0;
 if(FAULT==2)force dut.local_cells.negative_ce_ff.Q=0;
 if(FAULT==3)force dut.local_cells.positive_ce_ff.Q=0;
 if(FAULT==4)force dut.local_cells.feedback_ff.Q=0;
 #50;rst=1;#1000;
 if(!p[54] || !p[118])$fatal(1,"not complete");
 $display("CELLRAW %h",p);held=p;#1000;
 if(p!==held)$fatal(1,"finite probe did not stop");
 rst=0;#100;
 if(p[54] || p[118])$fatal(1,"result not cleared");
 // A stopped DDR clock must not return stale PASS after common reset.
 @(negedge ddr);run_ddr=0;rst=1;#1000;
 if(p[127:64]!==0 || !p[54])$fatal(1,"stopped domain stale result");
 run_ddr=1;#1000;
 if(!p[118])$fatal(1,"restart result absent");
 $display("CELLRAW %h",p);
 $display("PASS CELL_PROBE finite=1 reset=1 stopped_domain=1 fault=%0d",FAULT);$finish;
end
initial begin #10000;$fatal(1,"cell timeout");end
endmodule
'''
FAULT_TAIL=r'''
reg[2:0] forced_r=0,forced_w=0;
reg[63:0] held_bad;
integer checks=0;
task clean;begin
 @(negedge clk);rst_n=0;source_r_enable=0;source_b_enable=0;
 release dut.rstate;release dut.wstate;
 repeat(3)@(negedge clk);rst_n=1;repeat(3)@(negedge clk);
 if(adapter_diagnostic[255] || adapter_diagnostic[191] || protocol_error)$fatal(1,"reset/idle");
end endtask
initial begin
 clean;
 // No error flag need be raised: reproduce the exact observation blind spot.
 for(integer s=1;s<=4;s=s+1)begin
  forced_r=s;force dut.rstate=forced_r;
  @(negedge clk);
  if(!adapter_diagnostic[255] || adapter_diagnostic[130+:3]===3'bxxx ||
     adapter_diagnostic[248+:3]!==forced_r ||
     adapter_diagnostic[194:192]!==0 || adapter_diagnostic[95])$fatal(1,"uncaught read transition");
  held_bad=adapter_diagnostic[255:192];repeat(6)@(negedge clk);
  if(adapter_diagnostic[255:192]!==held_bad)$fatal(1,"first read evidence overwritten");
  checks=checks+1;clean;
 end
 for(integer s=1;s<=4;s=s+1)begin
  forced_w=s;force dut.wstate=forced_w;@(negedge clk);
  if(!adapter_diagnostic[255] || adapter_diagnostic[251+:3]!==forced_w)$fatal(1,"uncaught write transition");
  checks=checks+1;clean;
 end
 // A sub-cycle asynchronous reset must clear the history too.
 #1;rst_n=0;#1;rst_n=1;#1;
 if(adapter_diagnostic[255:128]!==0)$fatal(1,"async reset history stale");
 forced_r=3;force dut.rstate=forced_r;@(negedge clk);
 if(!adapter_diagnostic[255] || !adapter_diagnostic[254])$fatal(1,"first released edge not recorded");
 checks=checks+1;clean;
 $display("PASS STARTUP_TRACE intentional_transitions=%0d no_protocol_error_needed=1 async_reset=1",checks);$finish;
end
endmodule
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--board',required=True,type=Path);p.add_argument('--root',required=True,type=Path);a=p.parse_args()
 board=a.board.resolve();root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
 audit_delta(board);root.mkdir()
 for n in ('sapphire_axi32_to_axi128.sv','bringup_clock_probe.sv','jtag_bringup_snapshot.sv','bringup_cell_probe.sv'):shutil.copy2(board/'assembly'/n,root/n)
 for n in ('efx_lut4.v','efx_comb4.v','efx_ff.v'):shutil.copy2(EF/'sim_models/verilog'/n,root/n)
 tb=(HERE.parent/'ddr_fabric/tb_sapphire_axi32_to_axi128.sv').read_text().replace('    ) dut(.*);','    ) dut(.*);\n    wire [255:0] adapter_diagnostic;')
 faults=tb[:tb.index('    // Byte-addressed memory')].replace('module tb_sapphire_axi32_to_axi128','module tb_trace_faults')+FAULT_TAIL
 monitor='''
always @(negedge clk)if(rst_n && adapter_diagnostic[255])$fatal(1,"normal transaction false transition alarm %h",adapter_diagnostic);
'''
 (root/'tb_adapter.sv').write_text(tb.replace('endmodule',monitor+'\nendmodule'))
 (root/'tb_trace_faults.sv').write_text(faults)
 serial=SERIAL_TB.replace('[703:0]','[959:0]').replace('[767:0]','[1023:0]').replace('[767:64]','[1023:64]')
 serial=serial.replace('i<768','i<1024').replace('k<22','k<30').replace("32'h4f465034","32'h4f465035").replace('!==4','!==5').replace('OFP4','OFP5').replace('bits=768','bits=1024')
 serial=serial.replace('#250000','#350000')
 (root/'tb_serial.sv').write_text(serial);(root/'tb_clock.sv').write_text(CLOCK_TB);(root/'tb_cells.sv').write_text(CELL_TB)
 inp=dict(board=str(board),board_inputs_sha256=sha(board/'inputs.json'),runner_sha256=sha(__file__),decoder_sha256=sha(HERE/'read_startup_cause.py'),files={f.name:sha(f) for f in root.iterdir()},hardware_access=False)
 (root/'inputs.json').write_text(json.dumps(inp,indent=2))
 env=os.environ.copy();env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 def run(args,name,budget=60):
  start=time.monotonic()
  with (root/name).open('x') as log:
   proc=subprocess.Popen([str(Q/args[0]),*args[1:]],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
   try:code=proc.wait(timeout=budget)
   except subprocess.TimeoutExpired:stop_launched_tree(proc);raise
  body=(root/name).read_text(errors='replace');assert code==0 and not re.search(r'(?m)^#?\s*\*\* (Error|Fatal)',body),(name,body[-3000:])
  return dict(log=name,seconds=time.monotonic()-start,sha256=sha(root/name),passes=re.findall(r'^# (PASS .*)$',body,re.M)),body
 try:
  run(['vlib.exe','work'],'vlib.log',15)
  run(['vlog.exe','-sv','-timescale','1ns/1ps',*[f.name for f in root.glob('*.v')],*[f.name for f in root.glob('*.sv')]],'compile.log',45)
  jobs=[('clock','tb_clock',[]),('serial','tb_serial',[]),('trace','tb_trace_faults',[])]
  for fault in range(6):jobs.append((f'cells{fault}','tb_cells',[f'-gFAULT={fault}']))
  for seed in (71,1027):
   for tr,bad in ((0,0),(1,0),(1,1)):jobs.append((f'adapter{seed}-{tr}-{bad}','tb_sapphire_axi32_to_axi128',[f'-gTRANSLATION={tr}',f'-gBAD_MAPPING={bad}',f'+SEED={seed}']))
  results=[]
  for name,top,opts in jobs:
   row,body=run(['vsim.exe','-c','work.'+top,*opts,'-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],name+'.log')
   assert row['passes'],row
   if name.startswith('cells'):
    raw=re.findall(r'^# CELLRAW ([0-9a-f]+)$',body,re.M);assert len(raw)==2
    row['decoded']=[dict(process150=cell_decode(int(v,16)&((1<<64)-1)),ddr400=cell_decode(int(v,16)>>64)) for v in raw]
    for d in row['decoded']:assert d['ddr400']['matches'] and d['process150']['matches']==(name in ('cells0','cells5')),d
   if name=='serial':
    raw=re.findall(r'^# OFP5RAW ([0-9a-f]+)$',body,re.M);assert len(raw)==2
    for v in raw:
     # Serial test intentionally uses arbitrary bits, including reserved legacy
     # control fields. Validate the decoder separately with those fields zero.
     valid_payload=int(v,16)&~(((1<<384)-1)<<64)
     assert decode(valid_payload<<2)['format']=='OFP5'
   results.append(row)
  for n,h in inp['files'].items():assert sha(root/n)==h
  result=dict(status='STARTUP_CAUSE_TRACE_CELLS_SERIAL_ADAPTER_PASS',results=results,inputs_sha256=sha(root/'inputs.json'),hardware_access=False,physical_fit=False,board_release=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
