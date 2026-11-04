"""Check optimized model lookup against untouched Micron memory API."""
import json,os,subprocess
from pathlib import Path
from run_units import ROOT,EF,sha
from prepare_board_frame_run import fast_memory
def main():
 root=ROOT/'tmp/frame-execution/ddr-memory-backend1';root.mkdir(exist_ok=False)
 src=EF/'ipm/ip/efx_ddr3_soft_controller/testbench/ddr3.v';original=src.read_text()
 (root/'ddr3.v').write_text(original);(root/'ddr3_fast.v').write_text(fast_memory(original).replace('module ddr3 (','module ddr3_fast (',1))
 import shutil
 shutil.copy2(ROOT/'tmp/frame-execution/ddr-unit-startup5/2048Mb_ddr3_parameters.vh',root/'2048Mb_ddr3_parameters.vh')
 (root/'ddr3_device_ID.vh').write_text('`define sg125\n')
 tb='''`timescale 1ns/1ps
module tb;
ddr3 #(.MEM_BITS(13),.DEBUG(0)) reference(.rst_n(1'b0),.ck(1'b0),.ck_n(1'b1),.cke(1'b0),.cs_n(1'b1),.ras_n(1'b1),.cas_n(1'b1),.we_n(1'b1),.ba(3'd0),.addr(14'd0),.odt(1'b0));
ddr3_fast #(.MEM_BITS(13),.DEBUG(0)) fast(.rst_n(1'b0),.ck(1'b0),.ck_n(1'b1),.cke(1'b0),.cs_n(1'b1),.ras_n(1'b1),.cas_n(1'b1),.we_n(1'b1),.ba(3'd0),.addr(14'd0),.odt(1'b0));
reg [127:0] a,b,data;reg [2:0] bank;reg [13:0] row;reg [9:0] col;
task compare(input integer index);
 begin bank=index%8;row=(index/8)%16384;col=(index*8)%1024;
 reference.memory_read(bank,row,col,a);fast.memory_read(bank,row,col,b);
 if(a!==b)$fatal(1,"Backend mismatch index=%0d a=%h b=%h",index,a,b);end
endtask
initial begin
 #1;
 for(integer k=0;k<3000;k=k+1)begin
  bank=k%8;row=(k/8)%16384;col=(k*8)%1024;data={32'(k),32'(k^32'habcde),32'(~k),32'(k*357)};
  reference.memory_write(bank,row,col,data);fast.memory_write(bank,row,col,data);
 end
 for(integer k=3000;k>=0;k=k-1)compare(k);
 // Existing entry overwrite and absent reads, followed by compaction on erase.
 reference.memory_write(2,90,16,128'h1234);fast.memory_write(2,90,16,128'h1234);
 reference.erase_banks(8'h55);fast.erase_banks(8'h55);
 for(integer k=0;k<3100;k=k+1)compare(k);
 reference.memory_write(0,0,0,128'hfedcba);fast.memory_write(0,0,0,128'hfedcba);compare(0);
 $display("PASS DDR_MEMORY_BACKEND 6102 comparisons overwrite erase unread");$finish;
end
endmodule
'''
 (root/'tb.sv').write_text(tb);sim=Path('E:/quatra/questa_fse/win64');env=os.environ.copy();env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 for stage,args in [('vlib',['vlib.exe','work']),('compile',['vlog.exe','-sv','+define+den2048Mb+sg125+x16','+incdir+.','ddr3.v','ddr3_fast.v','tb.sv']),('simulation',['vsim.exe','-c','work.tb','-do','onbreak {quit -code 1}; run -all; quit -code 0'])]:
  with(root/(stage+'.log')).open('wb')as log:subprocess.run([str(sim/args[0]),*args[1:]],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 assert 'PASS DDR_MEMORY_BACKEND' in(root/'simulation.log').read_text()
 (root/'result.json').write_text(json.dumps(dict(status='DDR_MODEL_LOOKUP_EQUIVALENCE_PASS',checks=6102,original_sha256=sha(root/'ddr3.v'),fast_sha256=sha(root/'ddr3_fast.v'),log_sha256=sha(root/'simulation.log')),indent=2))
 print('DDR_MODEL_LOOKUP_EQUIVALENCE_PASS',flush=True)
if __name__=='__main__':main()
