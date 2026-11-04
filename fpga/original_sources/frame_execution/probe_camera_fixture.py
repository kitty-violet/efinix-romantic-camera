"""Short real RAW10 TX -> frozen board CSI RX check, prior to full board run."""
import json,os,shutil,subprocess,time
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
def main():
 root=ROOT/'tmp/frame-execution/camera-fixture1';root.mkdir(exist_ok=False)
 tx=EF/'ipm/ip/efx_csi2/testbench/soft_dphy/rx/efx_csi2_tx_modelsim.sv'
 sources=[tx,ROOT/'tmp/frame-execution/board-startup3/csi_rx_controller_sim.sv',HERE/'board_integration/camera_raw10_fixture.sv',*sorted((EF/'sim_models/verilog').glob('*.v'))]
 header=sources[1].read_text().split('module csi_rx_controller',1)[1].split(');',1)[0]
 import re
 connections={'reset_n':'rst','reset_byte_HS_n':'rst','reset_pixel_n':'rst','clk':'clk','clk_byte_HS':'bc','clk_pixel':'pc',
 'Rx_LP_CLK_P':'cp','Rx_LP_CLK_N':'cn','Rx_LP_D_P':'dp','Rx_LP_D_N':'dn',**{'Rx_HS_D_'+str(i):'d'+str(i) for i in range(4)},
 'fifo_rd_empty':"4'd0",'pixel_data':'data','pixel_data_valid':'valid','hsync_vc0':'hs','vsync_vc0':'vs','mipi_debug_out':'debug'}
 ports=[]
 for direction,width,name in re.findall(r'\b(input|output)\s+(\[[^]]+\])?\s*(\w+)',header):
  ports.append('.'+name+'('+connections.get(name,"'0" if direction=='input' else '')+')')
 tb='''`timescale 1ns/1ps
module tb;
reg clk=0,bc=0,pc=0,rst=0;
always #5 clk=~clk;always #7.142857 bc=~bc;always #10 pc=~pc;
wire cp,cn;wire [3:0]dp,dn,he;wire [7:0]d0,d1,d2,d3;
wire [63:0]data;wire valid,hs,vs;wire [31:0]debug;
camera_raw10_fixture #(.LINES(4),.FRAMES(1),.FRAME_TICKS(10000)) cam(.clk(clk),.byte_clk(bc),.pixel_clk(pc),.reset_n(rst),.start(1'b1),
 .lp_cp(cp),.lp_cn(cn),.lp_dp(dp),.lp_dn(dn),.d0(d0),.d1(d1),.d2(d2),.d3(d3),.hs_enable(he));
'''+ 'csi_rx_controller rx('+','.join(ports)+''');
integer words=0;integer lines=0;reg prevhs=0;
always @(posedge pc)if(rst)begin
 if(hs && !prevhs)lines=lines+1;prevhs=hs;
 if(valid)begin
  if(!hs || !vs || data[39:0]!=={4{10'h200}})$fatal(1,"CSI mismatch word=%0d data=%h hs=%b vs=%b debug=%h",words,data,hs,vs,debug);
  words=words+1;
 end
end
initial begin #200;rst=1;#400000;
 $display("CSI_SUM words=%0d lines=%0d debug=%h",words,lines,debug);
 if(words!=1920 || lines!=4 || (debug&32'h00007ffc)!=0)$fatal(1,"CSI count/error");
 $display("PASS CSI_REAL_RAW10_4_LINES");$finish;
end
endmodule
'''
 (root/'tb.sv').write_text(tb);sources.append(root/'tb.sv')
 env=os.environ.copy();env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat';sim=Path('E:/quatra/questa_fse/win64')
 for stage,cmd in [('vlib',[sim/'vlib.exe','work']),('compile',[sim/'vlog.exe','-sv',*sources]),('simulation',[sim/'vsim.exe','-c','work.tb','-do','onbreak {quit -code 1}; run -all; quit -code 0'])]:
  with (root/(stage+'.log')).open('wb')as f:subprocess.run(list(map(str,cmd)),cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 body=(root/'simulation.log').read_text();assert 'PASS CSI_REAL_RAW10_4_LINES' in body
 (root/'result.json').write_text(json.dumps(dict(status='CSI_FIXTURE_PASS',log_sha256=sha(root/'simulation.log')),indent=2))
 print('CSI_FIXTURE_PASS',flush=True)
if __name__=='__main__':main()
