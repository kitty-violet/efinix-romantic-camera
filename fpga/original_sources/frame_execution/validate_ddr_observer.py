"""Four-state counters, phase masks, stopped clock and read-only APB checks."""
import argparse,json,os,shutil
from pathlib import Path
from run_units import ROOT,HERE,sha,run
from ddr_cal_observer import BASE,MODULE,transform_apb
Q=Path('E:/quatra/questa_fse/win64')
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root.resolve()
 assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
 shutil.copy2(MODULE,r/MODULE.name)
 (r/'cnn_apb_control.sv').write_text(transform_apb((BASE/'assembly/cnn_apb_control.sv').read_text()))
 tb=r'''`timescale 1ns/1ps
module tb;
reg clk=0,core=0,tdqss=0,tac=0,twd=0,core_run=1;
always #3.333 clk=~clk;
always #2.5 if(core_run)core=~core;
always #1.25 tdqss=~tdqss;
always #1.25 tac=~tac;
initial begin #0.625;forever #1.25 twd=~twd;end
reg sysn=1,ddrn=0,cke=0,ena=0;reg[2:0] code=0;reg[4:0] sel=0;
reg[15:0] address=16'h70;wire[31:0] data,apbdata;reg reset=1,pwrite=0;
wire ready,error;wire enable;
ddr_cal_observer dut(clk,core,tdqss,tac,twd,sysn,ddrn,cke,ena,code,sel,address,data);
cnn_apb_control control(.clk(clk),.reset(reset),.paddr(address),.psel(1'b1),.penable(1'b1),
 .pwrite(pwrite),.pwdata(32'hffffffff),.pready(ready),.prdata(apbdata),.pslverror(error),
 .ddr_observation_data(data),.enable(enable));
task read_word(input[15:0] a,output[31:0] v);begin
 @(negedge clk);address=a;pwrite=0;#1;
 if(!ready || error || $isunknown(apbdata))$fatal(1,"APB observation read invalid %h",a);
 v=apbdata;
end endtask
reg[31:0] v,c0,c1,c2,c3,t0,t1,before_stop,after_stop,events;integer n;
initial begin
 #100;reset=0;
 read_word(16'h0000,v);if(v!==32'h54493630)$fatal(1,"identity changed");
 read_word(16'h0070,v);if(v!==32'h44434f31)$fatal(1,"observer identity");
 // Actual input clocks advance the four independent divided counters.
 #512000;
 read_word(16'h0084,t0);read_word(16'h0074,c0);read_word(16'h0078,c1);
 read_word(16'h007c,c2);read_word(16'h0080,c3);
 if(c0<398 || c0>402 || c1<798 || c1>802 || c2!=c1 || c3+1<c1 || c3>c1+1)
  $fatal(1,"clock counts %d %d %d %d",c0,c1,c2,c3);
 if(t0<298 || t0>302)$fatal(1,"timebase");
 // The divider cannot report activity after its clock has stopped.
 @(negedge core);core_run=0;#100;read_word(16'h0074,before_stop);
 #20000;read_word(16'h0074,after_stop);read_word(16'h0078,v);
 if(after_stop!=before_stop || v<=c1)$fatal(1,"stopped clock not isolated");core_run=1;
 // All eight phase codes; legitimate source-domain four-cycle enable pulses.
 for(n=0;n<8;n=n+1)begin
  @(negedge tdqss);code=n;sel=4;
  @(negedge tdqss);ena=1;
  repeat(4)@(negedge tdqss);
  ena=0;sel=0;repeat(4)@(negedge tdqss);
 end
 #100;read_word(16'h0090,v);
 if(v[19:12]!==8'hff || v[11:7]!==5'd4)$fatal(1,"phase/select seen %h",v);
 // Two complete DDR reset pulses, independent from functional reset.
 ddrn=1;cke=1;#30000;ddrn=0;cke=0;#40000;ddrn=1;cke=1;#100;
 read_word(16'h0088,events);if(events!==32'h00010002)$fatal(1,"DDR edge count %h",events);
 read_word(16'h008c,v);if(v[15:0]<22 || v[15:0]>25 || v[31:16]<16 || v[31:16]>19)$fatal(1,"reset widths %h",v);
 sysn=0;reset=1;#100;sysn=1;reset=0;#100;
 read_word(16'h0094,v);if(v[31:16]!==16'd1)$fatal(1,"system reset count");
 read_word(16'h0088,v);if(v!==events)$fatal(1,"functional reset lost diagnostic history");
 read_word(16'h0090,v);if(v[19:12]!==8'hff)$fatal(1,"functional reset cleared phases");
 for(n=16'h70;n<=16'h94;n=n+4)begin
  @(negedge clk);address=n;pwrite=1;#1;
  if(!ready || !error)$fatal(1,"diagnostic write accepted");
 end
 @(negedge clk);address=16'h71;pwrite=0;#1;if(!error)$fatal(1,"unaligned read accepted");
 @(negedge clk);address=16'h98;#1;if(!error || enable)$fatal(1,"unknown read/side effect");
 $display("PASS DDR_OBSERVER clocks=4 phases=8 stop=1 resets=2 readonly_words=10");$finish;
end
initial begin #1000000;$fatal(1,"simulation time limit");end
endmodule
'''
 (r/'tb.sv').write_text(tb)
 inputs=dict(module_sha256=sha(MODULE),transform_sha256=sha(HERE/'ddr_cal_observer.py'),
             parent_inputs_sha256=sha(BASE/'inputs.json'),files={f.name:sha(f) for f in r.iterdir()},runner_sha256=sha(__file__))
 (r/'inputs.json').write_text(json.dumps(inputs,indent=2))
 os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 run([Q/'vlib.exe','work'],r,'vlib.log',15)
 run([Q/'vlog.exe','-sv','ddr_cal_observer.sv','cnn_apb_control.sv','tb.sv'],r,'compile.log',25)
 text,seconds=run([Q/'vsim.exe','-c','work.tb','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],r,'simulation.log',60)
 assert 'PASS DDR_OBSERVER clocks=4 phases=8 stop=1 resets=2 readonly_words=10' in text
 for n,h in inputs['files'].items():assert sha(r/n)==h
 result=dict(status='DDR_OBSERVER_FOUR_STATE_SHORT_PASS',seconds=seconds,inputs_sha256=sha(r/'inputs.json'),
             log_sha256=sha(r/'simulation.log'),hardware_access=False)
 (r/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
