"""Positive and negative checks for early pin/AXI measurement diagnostics."""
import argparse,json,re,subprocess,time
from pathlib import Path
from run_units import ROOT,HERE,sha

TB='''`timescale 1ns/1ps
module tb #(parameter BAD_WTR=0,BAD_ADDR=0);
 reg ck=0,user_clk=0;always #1.25 ck=~ck;always #3.333 user_clk=~user_clk;
 reg [2:0] command=7;reg avalid=0;
 ddr516_measurement_guard dut(.user_clk(user_clk),.enabled(1'b1),.ck(ck),.reset_n(1'b1),
 .cke(1'b1),.cs_n(1'b0),.ras_n(command[2]),.cas_n(command[1]),.we_n(command[0]),
 .addr(14'b0),.bank(3'b0),.avalid(avalid),.aready(1'b1),.atype(1'b1),
 .aaddr(BAD_ADDR?32'hfff0:32'h1000),.aid(8'h3a),.alen(8'd1),.asize(3'd4),.aburst(2'b1));
 initial begin
  @(negedge user_clk);avalid=1;@(negedge user_clk);avalid=0;
  @(negedge ck);command=3'b100;
  @(negedge ck);command=7;
  repeat(BAD_WTR?11:12)@(negedge ck);command=3'b101;
  @(negedge ck);command=7;
  #10;$display("PASS MEASUREMENT_GUARD_BOUNDARY");$finish;
 end
endmodule
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
 source=HERE/'board_integration/ddr516_measurement_guard.sv'
 (root/source.name).write_bytes(source.read_bytes());(root/'tb.sv').write_text(TB)
 (root/'run.do').write_text('onerror {quit -code 1}\nonbreak {quit -code 1}\nrun -all\nquit -code 0\n')
 sim=Path('E:/altera/15.0/modelsim_ase/win32aloem');results=[]
 runs=[('vlib',[sim/'vlib.exe','work'],None),('compile',[sim/'vlog.exe','-sv',source.name,'tb.sv'],None)]
 for case,param,marker in [('normal',[],None),('short_gap',['-gBAD_WTR=1'],'BOARD_DDR_PIN_TWTR interval violation'),
                          ('cross_4k',['-gBAD_ADDR=1'],'BOARD_DDR_ADDRESS_SHAPE')]:
  runs.append((case,[sim/'vsim.exe','-c','work.tb',*param,'-do','run.do'],marker))
 for name,command,marker in runs:
  start=time.monotonic()
  with (root/(name+'.log')).open('wb') as f:r=subprocess.run(list(map(str,command)),cwd=root,stdout=f,stderr=subprocess.STDOUT,timeout=30)
  body=(root/(name+'.log')).read_text(errors='replace');error=bool(re.search(r'\*\* (?:Fatal|Error)',body))
  ok=(error and marker in body and 'PASS MEASUREMENT_GUARD' not in body) if marker else (r.returncode==0 and not error)
  if name=='normal':ok=ok and 'PASS MEASUREMENT_GUARD_BOUNDARY' in body
  if name=='short_gap':ok=ok and 'delta_ck=12 required_ck=13' in body and 'BOARD_DDR_PRIOR_A index=0' in body
  results.append(dict(case=name,passed=bool(ok),seconds=time.monotonic()-start,log_sha256=sha(root/(name+'.log'))))
  if not ok:break
 passed=len(results)==5 and all(row['passed'] for row in results)
 result=dict(status='MEASUREMENT_GUARD_PASS' if passed else 'MEASUREMENT_GUARD_FAIL',results=results,source_sha256=sha(source),full_board=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
 if not passed:raise SystemExit(1)
if __name__=='__main__':main()
