"""Reproduce and repair the public PLL model's simultaneous deassert corner.

Ti60 datasheet, Dynamic Phase Shift step 4, deasserts SHIFT_ENA and SHIFT_SEL
together. The official model captures PHASE_STEP while enabled but updates its
output delay only with SEL=1, ENA=0. Keep all clock-generation/PHY behavior;
apply the captured setting when enable is low, independently of current SEL.
This is a project simulation compatibility correction, not a silicon change.
"""
import argparse,json,os,re,shutil,subprocess,time
from pathlib import Path
from run_units import ROOT,EF,sha

def correct_model(original):
 result=original
 for k in range(5):
  pattern=rf'if \(SHIFT_SEL\[{k}\]\)\s*if \(SHIFT_ENA\) CLKOUT{k}_PHASE_STEP_r = SHIFT;\s*else  CLKOUT{k}_shift = \(VCO_period \* O \* CLKOUT{k}_PHASE_STEP_r\) / 2;'
  change=f'''if (SHIFT_SEL[{k}] && SHIFT_ENA) CLKOUT{k}_PHASE_STEP_r = SHIFT;
      // Preserve captured code after simultaneous ENA/SEL deassertion.
      if (!SHIFT_ENA) CLKOUT{k}_shift = (VCO_period * O * CLKOUT{k}_PHASE_STEP_r) / 2;'''
  result,n=re.subn(pattern,change,result);assert n==1,(k,n)
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--v4',action='store_true');p.add_argument('--startup-reset',action='store_true');a=p.parse_args()
 assert not a.startup_reset or a.v4
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
 official=EF/'pt/sim_models/verilog/EFX_PLL_V3.v';shutil.copy2(official,root/official.name)
 fixed=root/'EFX_PLL_V3_PROTOCOL.v'
 fixed.write_text(correct_model(official.read_text()).replace('module EFX_PLL_V3(','module EFX_PLL_V3_PROTOCOL(',1))
 old_fast=ROOT/'tmp/frame-execution/ddr516-pll-equivalence1/EFX_PLL_V3_FAST.v'
 fast=root/'EFX_PLL_V3_FAST_PROTOCOL.v'
 fast.write_text(correct_model(old_fast.read_text()).replace('module EFX_PLL_V3_FAST(','module EFX_PLL_V3_FAST_PROTOCOL(',1))
 if a.startup_reset:
  source=fast.read_text();start=source.index('   // Simulation optimization,');end=source.index('   // Calculate new shift values',start)
  replacement='''   // Preserve the official pre-lock VCO wakeup grid during initial reset.
   // After the first lock the stable-reference output counters are analytical.
   // Reset after lock is deliberately rejected; this warm measurement does not
   // claim arbitrary reset/relock support from an optimized model.
   reg once_locked=0;
   always @(posedge LOCKED) once_locked=1;
   always @(negedge RSTN_w) if(once_locked) $fatal(1,"Fast PLL reset after lock unsupported");
   always @(posedge LOCKED or negedge RSTN_w) begin
      while(!LOCKED)begin VCO=0;#(VCO_period/2);end
      VCO=1;
   end
'''
  for k in range(5):
   replacement+=f'''   always @(posedge VCO) begin : fixed_output_{k}
      time half_vco;
      half_vco=VCO_period/2;CLKOUT{k}_r=1;
      forever begin #(half_vco * O * CLKOUT{k}_DIV);CLKOUT{k}_r=~CLKOUT{k}_r;end
   end
'''
  fast.write_text(source[:start]+replacement+source[end:])
 inst='''#(.N(1),.M(1),.O(2),.CLKOUT0_DIV(6),.CLKOUT1_DIV(12),.CLKOUT2_DIV(6),.CLKOUT3_DIV(6),.CLKOUT4_DIV(48),
 .CLKOUT3_PHASE_STEP(3),.CLKOUT2_DYNPHASE_EN(1),.FEEDBACK_CLK("CLK4"),.FEEDBACK_MODE("CORE"),.REFCLK_FREQ(50.0))'''
 tb='''`timescale 1fs/1fs
module tb;
reg ref50=0;always #10000000 ref50=~ref50;
reg [2:0] shift=0;reg [4:0] sel=0;reg ena=0;
wire [4:0] a,b,c;wire la,lb,lc;
integer checks=0,corner_reproduced=0,phase_checks=0;reg check_phase=0;
time last_base=0;time observed=0;time expected=0;
'''
 for module,name,clks,lock in [('EFX_PLL_V3','original','a','la'),('EFX_PLL_V3_PROTOCOL','corrected','b','lb'),('EFX_PLL_V3_FAST_PROTOCOL','fast','c','lc')]:
  tb+=module+' '+inst+' '+name+'''(.CLKIN({ref50,3'bzzz}),.CLKSEL(2'd3),.RSTN(1'b1),
 .SHIFT(shift),.SHIFT_SEL(sel),.SHIFT_ENA(ena),'''+f'.FBK({clks}[4]),.LOCKED({lock}),'+','.join(f'.CLKOUT{k}({clks}[{k}])' for k in range(5))+');\n'
 tb+='''always @(posedge b[0])last_base=$time;
always @(posedge b[2])begin
 #1;
 if(check_phase)begin
  observed=$time-1-last_base;
  if(observed+1<expected || observed>expected+1)$fatal(1,"PLL output edge wrong actual=%0d expected=%0d",observed,expected);
  phase_checks=phase_checks+1;
 end
end
for(genvar k=0;k<5;k=k+1)begin
 always @(b[k] or c[k])begin #1;
  if(b[k]!==c[k])$fatal(1,"Reduced PLL mismatch output %0d",k);
  if($time>90000001 && $isunknown(b[k]))$fatal(1,"Unknown PLL output");checks=checks+1;
 end
 if(k!=2)always @(a[k] or b[k])begin #1;
  if(a[k]!==b[k])$fatal(1,"Unselected clock changed %0d",k);
 end
end
initial begin
 #107000003;
 for(integer code=0;code<8;code=code+1)begin
  check_phase=0;shift=code;#10000000;sel=4;ena=1;#48000000;
  sel=0;ena=0;#20000000;
  expected=corrected.VCO_period*corrected.O*code/2;
  if(corrected.CLKOUT2_shift<expected-0.51 || corrected.CLKOUT2_shift>expected+0.51)$fatal(1,"Setting not applied");
  if(code!=0)begin
   if(original.CLKOUT2_shift!=0)$fatal(1,"Original model corner not reproduced");
   corner_reproduced=corner_reproduced+1;
  end
  check_phase=1;#50000000;check_phase=0;
 end
 // Existing held-selection protocol must continue to work unchanged.
 for(integer code=0;code<8;code=code+1)begin
  shift=code;#10000000;sel=4;ena=1;#48000000;ena=0;#20000000;
  if(original.CLKOUT2_shift!=corrected.CLKOUT2_shift)$fatal(1,"Held-selection behavior changed");
  sel=0;#20000000;
 end
 if(corner_reproduced!=7 || phase_checks<150)$fatal(1,"Insufficient coverage");
 $display("PASS PLL_PROTOCOL_CORRECTION original_missed=%0d phase_checks=%0d fast_checks=%0d",corner_reproduced,phase_checks,checks);$finish;
end
endmodule
'''
 if a.v4:
  v4='''#(.N(1),.M(2),.O(2),.CLKOUT0_DIV(6),.CLKOUT1_DIV(6),.CLKOUT2_DIV(6),.CLKOUT3_DIV(12),.CLKOUT4_DIV(48),
 .CLKOUT0_PHASE_STEP(3),.CLKOUT2_DYNPHASE_EN(1),.FEEDBACK_CLK("CLK4"),.FEEDBACK_MODE("CORE"),.REFCLK_FREQ(25.0))'''
  tb=tb.replace(inst,v4).replace('#10000000 ref50=~ref50','#20000000 ref50=~ref50')
  tb=tb.replace('posedge b[0]','posedge b[1]').replace('#107000003;','#207000003;').replace('$time>90000001','$time>190000001')
 if a.startup_reset:
  tb=tb.replace('reg ref50=0;','reg reset_n=0;initial #100000000 reset_n=1;\nreg ref50=0;')
  tb=tb.replace('.RSTN(1\'b1)','.RSTN(reset_n)')
  tb=tb.replace('#207000003;','#307000003;').replace('$time>190000001','$time>290000001')
 (root/'tb.sv').write_text(tb);(root/'run.do').write_text('onerror {quit -code 1}\nonbreak {quit -code 1}\nrun -all\nquit -code 0\n')
 sim=Path('E:/quatra/questa_fse/win64');env=os.environ.copy();env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat';start=time.monotonic()
 for name,cmd in [('vlib',[sim/'vlib.exe','work']),('compile',[sim/'vlog.exe','-sv',official.name,fixed.name,fast.name,'tb.sv']),('simulation',[sim/'vsim.exe','-c','work.tb','-do','run.do'])]:
  with (root/(name+'.log')).open('wb') as f:r=subprocess.run(list(map(str,cmd)),cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=60)
  body=(root/(name+'.log')).read_text(errors='replace')
  if r.returncode or re.search(r'\*\* (?:Error|Fatal)',body):raise RuntimeError(body[-3500:])
 marker=re.search(r'PASS PLL_PROTOCOL_CORRECTION original_missed=(\d+) phase_checks=(\d+) fast_checks=(\d+)',body);assert marker
 result=dict(status='PLL_PROTOCOL_CORRECTION_PASS',original_missed=int(marker[1]),phase_checks=int(marker[2]),fast_checks=int(marker[3]),seconds=time.monotonic()-start,
  v4_clock_plan=a.v4,startup_reset=a.startup_reset,
  official_sha256=sha(official),corrected_sha256=sha(fixed),fast_corrected_sha256=sha(fast),log_sha256=sha(root/'simulation.log'),board_release=False,
  scope=('V4 25MHz plan with actual initial 100ns reset; later reset/relock and variable reference excluded' if a.startup_reset else
         'project simulation correction of public PLL model; fixed DDR clock plan, no PLL reset/variable reference proof'))
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
