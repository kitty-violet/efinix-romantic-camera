"""Actual public JTAG -> SoC decoder -> APB chain, without force or AXI memory."""
import argparse,json,os,shutil
from pathlib import Path
from run_units import ROOT,HERE,sha,run
from ddr_cal_observer import BASE,transform_soc,transform_apb
from validate_fixed_control_soc import DIRECT_TASKS
Q=Path('E:/quatra/questa_fse/win64')
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root.resolve()
 assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
 for name,fn in (('EfxSapphireSoc.v',transform_soc),('cnn_apb_control.sv',transform_apb)):
  (r/name).write_text(fn((BASE/'assembly'/name).read_text()))
 shutil.copy2(BASE/'assembly/jtag_bringup_snapshot.sv',r/'jtag_bringup_snapshot.sv')
 tasks=(HERE/'board_integration/jtag_memory_tasks.svh').read_text().split('task jdebug(')[0]+DIRECT_TASKS
 tb=r'''`timescale 1ns/1ps
module tb;
reg clk=0,reset=1,jtck=0,jtdi=0,jsel=0,jcapture=0,jshift=0,jupdate=0;
wire jtdo,sys_reset;wire[15:0] address;wire[31:0] wdata,rdata;wire psel,penable,pwrite,ready,error,enable;
wire awvalid,wvalid,arvalid;integer apb_count=0;
always #3.333 clk=~clk;
cnn_fixed_control_soc dut(.diagnostic_board_status(0),.diagnostic_levels(7'h7f),
 .io_systemClk(clk),.io_asyncReset(reset),.io_systemReset(sys_reset),
 .jtagCtrl_tck(jtck),.jtagCtrl_tdi(jtdi),.jtagCtrl_enable(jsel),.jtagCtrl_capture(jcapture),
 .jtagCtrl_shift(jshift),.jtagCtrl_update(jupdate),.jtagCtrl_reset(reset),.jtagCtrl_tdo(jtdo),
 .io_apbSlave_0_PADDR(address),.io_apbSlave_0_PSEL(psel),.io_apbSlave_0_PENABLE(penable),
 .io_apbSlave_0_PWRITE(pwrite),.io_apbSlave_0_PWDATA(wdata),.io_apbSlave_0_PRDATA(rdata),
 .io_apbSlave_0_PREADY(ready),.io_apbSlave_0_PSLVERROR(error),
 .axiA_awvalid(awvalid),.axiA_wvalid(wvalid),.axiA_arvalid(arvalid),
 .axiA_awready(1'b0),.axiA_wready(1'b0),.axiA_bvalid(1'b0),.axiA_arready(1'b0),.axiA_rvalid(1'b0));
// Distinct deterministic words prove address/response routing, not live counters.
wire[31:0] observation=address==16'h70 ? 32'h44434f31 : 32'ha5000000|address;
cnn_apb_control apb(.clk(clk),.reset(reset||sys_reset),.paddr(address),.psel(psel),.penable(penable),
 .pwrite(pwrite),.pwdata(wdata),.prdata(rdata),.pready(ready),.pslverror(error),.enable(enable),
 .ddr_observation_data(observation),.bringup_ddr_status(32'h1234abcd));
always @(posedge clk)if(!reset&&!sys_reset)begin
 if(awvalid||wvalid||arvalid||enable)$fatal(1,"read-only test caused DDR/run traffic");
 if(psel&&penable&&ready)apb_count=apb_count+1;
end
@@TASKS@@
integer i,initial_apb_count;reg[31:0] expected;
initial begin
 #1000;reset=0;#40000;initial_apb_count=apb_count;
 jexpect(32'hf8100000,32'h54493630);jexpect(32'hf810006c,32'h1234abcd);
 for(i=16'h70;i<=16'h94;i=i+4)begin
  expected=i==16'h70 ? 32'h44434f31 : 32'ha5000000|i;
  jexpect(32'hf8100000+i,expected);jreject(1,32'hf8100000+i,2);
 end
 jreject(0,32'hf8100098,2);jreject(0,32'hf8100071,2);jreject(0,32'hf8100070,1);
 jreject(0,32'hf8100170,2);jreject(0,32'hf8110070,2);jreject(1,32'hf8100008,2);
 if(apb_count-initial_apb_count!=12)$fatal(1,"rejected request reached APB count=%0d",apb_count-initial_apb_count);
 reset=1;#1000;reset=0;#40000;jexpect(32'hf8100070,32'h44434f31);
 $display("PASS DDR_OBSERVER_REAL_JTAG reads=13 rejected=16 reset=1 ddr_requests=0");$finish;
end
initial begin #1000000;$fatal(1,"bounded test timeout");end
endmodule
'''.replace('@@TASKS@@',tasks)
 (r/'tb.sv').write_text(tb)
 inputs=dict(files={f.name:sha(f) for f in r.iterdir()},transform_sha256=sha(HERE/'ddr_cal_observer.py'),runner_sha256=sha(__file__),hardware_access=False)
 (r/'inputs.json').write_text(json.dumps(inputs,indent=2));os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 run([Q/'vlib.exe','work'],r,'vlib.log',15)
 run([Q/'vlog.exe','-sv','EfxSapphireSoc.v','jtag_bringup_snapshot.sv','cnn_apb_control.sv','tb.sv'],r,'compile.log',30)
 text,seconds=run([Q/'vsim.exe','-c','work.tb','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],r,'simulation.log',60)
 assert 'PASS DDR_OBSERVER_REAL_JTAG reads=13 rejected=16 reset=1 ddr_requests=0' in text
 result=dict(status='DDR_OBSERVER_REAL_JTAG_APB_PASS',inputs_sha256=sha(r/'inputs.json'),log_sha256=sha(r/'simulation.log'),seconds=seconds,hardware_access=False)
 (r/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
