"""Prove a simulation-only fixed-reference PLL event reduction against Efinix.

Keep the official lock, phase update and transport-delay logic. Replace only
unused VCO/divider intermediate events with identical rounded output periods.
Restricted to the stable-reference, reset-tied-high DDR reference fixture.
"""
import argparse,json,os,re,shutil,subprocess,time
from pathlib import Path
from run_units import ROOT,EF,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
 official=EF/'pt/sim_models/verilog/EFX_PLL_V3.v';shutil.copy2(official,root/official.name)
 original=official.read_text();start=original.index('   // Generate the VCO clock');end=original.index('   // Calculate new shift values',start)
 reduced='''   // Simulation optimization, fixed reference and reset tied high only.
   always @(negedge RSTN_w) $fatal(1,"Fast reference PLL requires reset tied high");
'''
 for k in range(5):
  reduced+=f'''   always @(posedge LOCKED) begin : fixed_output_{k}
      time half_vco;
      half_vco = VCO_period/2;
      CLKOUT{k}_r = 1;
      forever begin #(half_vco * O * CLKOUT{k}_DIV); CLKOUT{k}_r = ~CLKOUT{k}_r; end
   end
'''
 optimized=(original[:start]+reduced+original[end:]).replace('module EFX_PLL_V3(','module EFX_PLL_V3_FAST(',1)
 fast=root/'EFX_PLL_V3_FAST.v';fast.write_text(optimized)
 inst='''#(.N(1),.M(1),.O(2),.CLKOUT0_DIV(6),.CLKOUT1_DIV(12),
 .CLKOUT2_DIV(6),.CLKOUT3_DIV(6),.CLKOUT4_DIV(48),.CLKOUT3_PHASE_STEP(3),
 .CLKOUT2_DYNPHASE_EN(1),.FEEDBACK_CLK("CLK4"),.FEEDBACK_MODE("CORE"),.REFCLK_FREQ(50.0))'''
 tb='''`timescale 1fs/1fs
module tb;
reg ref50=0;always #10000000 ref50=~ref50;
reg [2:0] shift=0;reg [4:0] sel=4;reg ena=0;
wire [4:0] a,b;wire la,lb;integer checks[0:4];integer total=0;
'''
 for module,name,clks,lock in [('EFX_PLL_V3','reference','a','la'),('EFX_PLL_V3_FAST','reduced','b','lb')]:
  tb+=module+' '+inst+' '+name+'''(.CLKIN({ref50,3'bzzz}),.CLKSEL(2'd3),.RSTN(1'b1),
 .SHIFT(shift),.SHIFT_SEL(sel),.SHIFT_ENA(ena),'''+f'.FBK({clks}[4]),.LOCKED({lock}),'+','.join(f'.CLKOUT{k}({clks}[{k}])' for k in range(5))+');\n'
 tb+='''for(genvar k=0;k<5;k=k+1)begin
 initial checks[k]=0;
 always @(a[k] or b[k])begin
  #1;
  if(a[k]!==b[k])$fatal(1,"PLL edge mismatch output=%0d t_fs=%0t ref=%b fast=%b",k,$time,a[k],b[k]);
  if($time>90000001 && $isunknown(a[k]))$fatal(1,"PLL unknown output");
  checks[k]=checks[k]+1;
 end
end
always @(la or lb)begin #1;if(la!==lb)$fatal(1,"PLL lock mismatch");end
initial begin
 #107000003;
 for(integer j=0;j<400;j=j+1)begin
  shift=j%8;sel=(j%9==8)?0:4;ena=1;#17000001;ena=0;#33000003;
 end
 #100000000;
 for(integer k=0;k<5;k=k+1)total=total+checks[k];
 if(total<40000)$fatal(1,"PLL insufficient checks %0d",total);
 $display("PASS DDR516_PLL_EDGE_EQUIVALENCE checks=%0d phase_updates=400",total);$finish;
end
endmodule
'''
 (root/'tb.sv').write_text(tb);(root/'run.do').write_text('onerror {quit -code 1}\nonbreak {quit -code 1}\nrun -all\nquit -code 0\n')
 sim=Path('E:/quatra/questa_fse/win64');env=os.environ.copy();env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat';start=time.monotonic()
 for name,cmd in [('vlib',[sim/'vlib.exe','work']),('compile',[sim/'vlog.exe','-sv',official.name,fast.name,'tb.sv']),('simulation',[sim/'vsim.exe','-c','work.tb','-do','run.do'])]:
  with (root/(name+'.log')).open('wb') as f:r=subprocess.run(list(map(str,cmd)),cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=60)
  body=(root/(name+'.log')).read_text(errors='replace')
  if r.returncode or re.search(r'\*\* (?:Error|Fatal)',body):raise RuntimeError(body[-3000:])
 marker=re.search(r'PASS DDR516_PLL_EDGE_EQUIVALENCE checks=(\d+) phase_updates=400',body);assert marker
 result=dict(status='FIXED_REFERENCE_PLL_EQUIVALENCE_PASS',checks=int(marker[1]),phase_updates=400,official_sha256=sha(official),fast_sha256=sha(fast),log_sha256=sha(root/'simulation.log'),seconds=time.monotonic()-start,scope='only tested fixed DDR clock plan; RSTN=1, stable 50MHz reference',board_release=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
