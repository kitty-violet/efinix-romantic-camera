"""Four-state first-fault record and actual transformed-RTL compile gate."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,EF,MS,sha,run
from first_fault_observer import BASE,MODULE,transform_soc,transform_observer
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args();r=a.root.resolve()
 assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
 (r/'EfxSapphireSoc.v').write_text(transform_soc((BASE/'assembly/EfxSapphireSoc.v').read_text()),encoding='utf-8')
 (r/'jtag_bringup_snapshot.sv').write_text(transform_observer((BASE/'assembly/jtag_bringup_snapshot.sv').read_text()),encoding='utf-8')
 tb=r'''`timescale 1ns/1ps
module tb;
reg clk=0,reset=1,fatal=0,boot_failed=0;reg [9:0] state=1;reg [5:0] boot=0;
reg [31:0] address=0,flags=0;reg [12:0] age=0;wire [191:0] data;
control_first_fault dut(clk,reset,state,boot,address,flags,age,fatal,boot_failed,data);
always #3.333 clk=~clk;
task tick;begin @(posedge clk);#1;end endtask
task restart;begin @(negedge clk);reset=1;tick();@(negedge clk);reset=0;end endtask
reg [191:0] first;integer tested=0;
initial begin
 tick();@(negedge clk);reset=0;
 // Every legal state, changing flags/address: no false record.
 for(integer i=0;i<100;i=i+1)begin
  state=10'b1<<(i%10);boot=i%56;address=i*4;flags=i;age=0;tick();
  if(data[63]!==0 || data[62]!==1)$fatal(1,"false trigger");@(negedge clk);
 end
 // Capture exactly the previous/current cycle, without compressed-state loss.
 state=10'h002;boot=6'd7;address=32'hf9000f00;flags=32'h12345678;tick();@(negedge clk);
 state=0;address=32'hf8100000;flags=32'h87654321;tick();
 if(data[63:58]!==6'b110001 || data[57:52]!==6'd7 || data[51:42]!==10'h002 || data[41:32]!==0 ||
    data[95:64]!==32'hf8100000 || data[127:96]!==32'h87654321 || data[159:128]!==32'h12345678 || data[191:160]!==32'hf9000f00)
  $fatal(1,"first event wrong %h",data);
 first=data;
 for(integer i=0;i<50;i=i+1)begin @(negedge clk);state=3;flags=i;address=i;tick();if(data!==first)$fatal(1,"record overwritten");end
 tested=tested+1;
 // Reset clears the previous record, and each independent trigger has its bit.
 for(integer kind=0;kind<3;kind=kind+1)begin
  state=1;flags=0;address=0;fatal=0;boot_failed=0;age=0;restart();tick();
  if(data[63]!==0)$fatal(1,"record not reset");
  @(negedge clk);
  if(kind==0)age=64;else if(kind==1)fatal=1;else boot_failed=1;
  tick();
  if(data[63]!==1 || data[61:58] !== (4'b0010<<kind))$fatal(1,"reason %0d %h",kind,data);
  tested=tested+1;
 end
 $display("PASS FIRST_FAULT_RECORD cases=%0d legal_cycles=100 held_cycles=50",tested);$finish;
end
initial begin #10000;$fatal(1,"test timeout");end
endmodule
'''
 (r/'tb.sv').write_text(tb)
 inputs=dict(parent_inputs_sha256=sha(BASE/'inputs.json'),module_sha256=sha(MODULE),transform_sha256=sha(HERE/'first_fault_observer.py'),
             files={p.name:sha(p) for p in r.iterdir() if p.is_file()},runner_sha256=sha(__file__))
 (r/'inputs.json').write_text(json.dumps(inputs,indent=2))
 run([MS/'vlib.exe','work'],r,'vlib.log',15)
 run([MS/'vlog.exe','-sv','-timescale','1ns/1ps','EfxSapphireSoc.v','jtag_bringup_snapshot.sv','tb.sv'],r,'compile.log',20)
 text,seconds=run([MS/'vsim.exe','-c','work.tb','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],r,'simulation.log',30)
 assert 'PASS FIRST_FAULT_RECORD cases=4' in text
 result=dict(status='FIRST_FAULT_FOUR_STATE_SHORT_PASS',seconds=seconds,inputs_sha256=sha(r/'inputs.json'),log_sha256=sha(r/'simulation.log'),hardware_access=False)
 (r/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
