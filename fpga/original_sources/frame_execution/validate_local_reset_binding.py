"""Short exact-binding test for independent 150MHz/5MHz reset release."""
import argparse,json,os,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,sha,run
from prepare_local_reset_board import audit_delta
MS=Path('E:/quatra/questa_fse/win64')


def main():
    p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    b=a.board.resolve();r=a.root.resolve();assert not r.exists();audit_delta(b);r.mkdir()
    top=b/'rtl/ti60f225_oob_top.v';source=top.read_text()
    start=source.index('board_startup_reset startup_reset (')
    end=source.index(');',source.index('board_startup_reset startup_reset_5m ('))+2
    binding=source[start:end]
    counter=re.search(r"reg \[12:0\] i2c_rst_cnt.*?wire i2c_rst_n = i2c_rst_cnt\[12\];",source,re.S)[0]
    assert 'w_arstn' not in counter and counter.count('w_i2c_arstn')==2
    tb='''`timescale 1ns/1ps
module tb;
 reg process_clk_150=0,CLK_5M=0,fast_run=1,slow_run=1;
 always #3.333 if(fast_run)process_clk_150=!process_clk_150;
 always #100 if(slow_run)CLK_5M=!CLK_5M;
 reg i_arstn=1;reg [3:0] locks=7;
 wire i_pll_locked=locks[0],pll_locked=locks[1],user_pll_locked=locks[2],process_pll_locked=locks[3];
 wire w_arstn;
 @@INSTANCES@@
 @@COUNTER@@
 integer fast_edges=0,slow_edges=0,checks=0;
 wire ready=i_arstn && (&locks);
 always @(negedge ready)begin fast_edges=0;slow_edges=0;#0.01;
  if(w_arstn!==0 || w_i2c_arstn!==0)$fatal(1,"direct async assertion");end
 always @(posedge process_clk_150)begin
  if(ready)fast_edges=fast_edges+1;else fast_edges=0;
  #0.01;if(w_arstn !== (ready && fast_edges>=66))$fatal(1,"fast edge %d",fast_edges);checks=checks+1;
 end
 always @(posedge CLK_5M)begin
  if(ready)slow_edges=slow_edges+1;else slow_edges=0;
  #0.01;if(w_i2c_arstn !== (ready && slow_edges>=66))$fatal(1,"slow edge %d",slow_edges);checks=checks+1;
  if(!w_i2c_arstn && i2c_rst_cnt!==0)$fatal(1,"counter left reset early");
 end
 task settle;begin wait(w_arstn===1 && w_i2c_arstn===1);repeat(3)@(negedge CLK_5M);end endtask
 initial begin
  repeat(100)@(negedge process_clk_150);#0.2;locks=15;settle();
  for(integer k=0;k<4;k=k+1)begin
   #17.23;locks[k]=0;#217.7;locks[k]=1;settle();
  end
  #31;i_arstn=0;#900;i_arstn=1;settle();
  // A stopped destination cannot release merely because the fast side ran.
  @(negedge CLK_5M);slow_run=0;#0.2;i_arstn=0;#37;i_arstn=1;
  repeat(200)@(negedge process_clk_150);
  if(w_arstn!==1 || w_i2c_arstn!==0)$fatal(1,"slow reset depended on fast release");
  slow_run=1;settle();
  // The converse: slow reset is qualified locally, never by a fast FF.
  @(negedge process_clk_150);fast_run=0;#0.2;i_arstn=0;#37;i_arstn=1;
  repeat(70)@(negedge CLK_5M);
  if(w_arstn!==0 || w_i2c_arstn!==1)$fatal(1,"fast/slow reset coupling");
  fast_run=1;settle();
  $display("PASS LOCAL_RESET_EXACT_BINDING checks=%0d fourlocks stoppedclocks manual i2c",checks);$finish;
 end
 initial begin #300000;$fatal(1,"timeout");end
endmodule
'''.replace('@@INSTANCES@@',binding).replace('@@COUNTER@@',counter)
    (r/'tb.sv').write_text(tb);src=b/'assembly/board_startup_reset.sv';shutil.copy2(src,r/src.name)
    meta=dict(board=str(b),top_sha256=sha(top),module_sha256=sha(src),binding=binding,counter=counter,
              testbench_sha256=sha(r/'tb.sv'),runner_sha256=sha(__file__),hardware_access=False)
    (r/'inputs.json').write_text(json.dumps(meta,indent=2))
    os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
    run([MS/'vlib.exe','work'],r,'vlib.log',timeout=25)
    body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',r/src.name,r/'tb.sv'],r,'compile.log',timeout=30)
    assert 'Errors: 0, Warnings: 0' in body
    body,seconds=run([MS/'vsim.exe','-c','work.tb','-do','onerror {quit -code 1}; run -all; quit -code 0'],r,'simulation.log',timeout=40)
    assert 'PASS LOCAL_RESET_EXACT_BINDING' in body and 'Errors: 0, Warnings: 0' in body
    result=dict(status='LOCAL_RESET_EXACT_TOP_BINDING_PASS',seconds=seconds,inputs_sha256=sha(r/'inputs.json'),
        top_sha256=sha(top),module_sha256=sha(src),simulation_sha256=sha(r/'simulation.log'),
        local_clocks=['process_clk_150','CLK_5M'],board_release=False)
    (r/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
