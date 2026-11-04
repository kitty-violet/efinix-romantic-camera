"""Exercise the exact generated top reset binding, including the fourth PLL."""
import argparse,json,os,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,sha,run
from prepare_four_pll_reset_board import audit_delta
MS=Path('E:/quatra/questa_fse/win64')


def main():
    p=argparse.ArgumentParser();p.add_argument('--board',required=True,type=Path)
    p.add_argument('--root',required=True,type=Path);a=p.parse_args()
    b=a.board.resolve();r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists()
    audit_delta(b);r.mkdir()
    top=b/'rtl/ti60f225_oob_top.v';source=top.read_text()
    binding=re.search(r'board_startup_reset startup_reset \(.*?\n\);',source,re.S)[0]
    assert '.plls_locked(i_pll_locked & pll_locked & user_pll_locked & process_pll_locked)' in binding
    tb='''`timescale 1ns/1ps
module tb_fourpll_reset_binding;
 reg process_clk_150=0;always #3.333 process_clk_150=!process_clk_150;
 reg i_arstn=1;reg [3:0] locks=4'b0111;
 wire i_pll_locked=locks[0],pll_locked=locks[1],user_pll_locked=locks[2],process_pll_locked=locks[3];
 wire w_arstn;
 @@RESET_INSTANCE@@
 task release_check;
  begin
   for(integer n=1;n<=66;n=n+1)begin
    @(posedge process_clk_150);#0.1;
    if(w_arstn !== (n==66))$fatal(1,"release edge %d",n);
   end
  end
 endtask
 initial begin
  // Three historic PLLs are locked but the processing clock PLL is not.
  repeat(100)begin @(negedge process_clk_150);if(w_arstn!==0)$fatal(1,"fourth PLL ignored");end
  locks=15;release_check();
  for(integer k=0;k<4;k=k+1)begin
   @(negedge process_clk_150);locks[k]=0;#0.1;
   if(w_arstn!==0)$fatal(1,"PLL loss not asserted %d",k);
   repeat(5)@(negedge process_clk_150);locks[k]=1;release_check();
  end
  @(negedge process_clk_150);i_arstn=0;#0.1;
  if(w_arstn!==0)$fatal(1,"manual reset not direct");
  repeat(5)@(negedge process_clk_150);i_arstn=1;release_check();
  $display("PASS FOURPLL_EXACT_TOP_RESET_BINDING all_four_locks manual_reset");$finish;
 end
 initial begin #20000;$fatal(1,"timeout");end
endmodule
'''.replace('@@RESET_INSTANCE@@',binding)
    (r/'tb.sv').write_text(tb)
    src=b/'assembly/board_startup_reset.sv';shutil.copy2(src,r/src.name)
    meta=dict(board=str(b),top_sha256=sha(top),module_sha256=sha(src),binding=binding,
              testbench_sha256=sha(r/'tb.sv'),runner_sha256=sha(__file__),hardware_access=False)
    (r/'inputs.json').write_text(json.dumps(meta,indent=2))
    os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
    run([MS/'vlib.exe','work'],r,'vlib.log',timeout=30)
    body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',r/src.name,r/'tb.sv'],r,'compile.log',timeout=40)
    assert 'Errors: 0, Warnings: 0' in body
    body,seconds=run([MS/'vsim.exe','-c','work.tb_fourpll_reset_binding',
        '-do','onerror {quit -code 1}; run -all; quit -code 0'],r,'simulation.log',timeout=45)
    assert 'PASS FOURPLL_EXACT_TOP_RESET_BINDING' in body and 'Errors: 0, Warnings: 0' in body
    result=dict(status='FOURPLL_EXACT_TOP_RESET_BINDING_PASS',seconds=seconds,
                inputs_sha256=sha(r/'inputs.json'),top_sha256=sha(top),module_sha256=sha(src),
                simulation_sha256=sha(r/'simulation.log'),board_release=False)
    (r/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
