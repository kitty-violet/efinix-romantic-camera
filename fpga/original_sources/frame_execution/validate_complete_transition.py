"""Bounded official-LUT simulation of all legal transitions/input combinations."""
import argparse,json,os,shutil,subprocess,time
from pathlib import Path
from run_units import ROOT,EF,sha
from complete_control_transition import audit_delta,circuit,REL
from windows_process_tree import stop_launched_tree

def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 board=a.board.resolve();root=a.root.resolve();audit_delta(board)
 assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
 text=(board/REL).read_text();start=text.index('module control_complete_next(')
 assert text[start:]==circuit()[0]
 (root/'control_complete_next.v').write_text(text[start:])
 shutil.copy2(EF/'sim_models/verilog/efx_lut4.v',root/'efx_lut4.v')
 tb=r'''`timescale 1ns/1ps
module tb;
reg [9:0] matched=0;
reg [12:0] stimulus=0;
wire h=stimulus[0],b=stimulus[1],bad=stimulus[2],mail=stimulus[3],apb=stimulus[4],writing=stimulus[5];
wire pr=stimulus[6],aw=stimulus[7],w=stimulus[8],bv=stimulus[9],ar=stimulus[10],rv=stimulus[11],last=stimulus[12];
wire [9:0] next_state;
control_complete_next dut(matched,h,b,bad,mail,apb,writing,pr,aw,w,bv,ar,rv,last,next_state);
integer state,inputs,checks=0,detected=0;
reg [9:0] expected;
task reference;
begin
 expected=matched;
 case(matched)
  1:if(h || b)expected=2;
  2:if(bad)expected=512;else if(mail)expected=writing ? 512:4;else if(apb)expected=8;else expected=writing ? 32:128;
  4:expected=512;
  8:expected=16;
  16:if(pr)expected=512;
  32:if(aw && w)expected=64;
  64:if(bv)expected=512;
  128:if(ar)expected=256;
  256:if(rv && last)expected=512;
  512:expected=1;
  default:expected=1;
 endcase
end endtask
initial begin
 for(state=0;state<11;state=state+1)begin
  matched=state==10 ? 0:(1<<state);
  for(inputs=0;inputs<8192;inputs=inputs+1)begin
   stimulus=inputs;#1;reference;
   if(next_state!==expected)$fatal(1,"Mismatch state=%h stimulus=%h expected=%h actual=%h",matched,stimulus,expected,next_state);
   checks=checks+1;
  end
 end
 // Check the exact observed bad transition is rejected by the same oracle.
 matched=8;stimulus=0;#1;reference;
 force dut.next_state[9]=1;#1;
 if(next_state!==expected)detected=detected+1;
 release dut.next_state[9];#1;
 if(next_state!==16 || checks!=90112 || detected!=1)$fatal(1,"Recovery/count/mutation rejection");
 $display("PASS COMPLETE_NEXT_STATE checks=%0d fault_rejected=%0d",checks,detected);$finish;
end
initial begin #100000;$fatal(1,"Simulation limit");end
endmodule
'''
 (root/'tb.sv').write_text(tb)
 inp=dict(board_inputs_sha256=sha(board/'inputs.json'),soc_sha256=sha(board/REL),
  files={n:sha(root/n) for n in ('control_complete_next.v','efx_lut4.v','tb.sv')},runner_sha256=sha(__file__),hardware_access=False)
 (root/'inputs.json').write_text(json.dumps(inp,indent=2))
 env=os.environ.copy();env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 q=Path('E:/quatra/questa_fse/win64');started=time.monotonic()
 try:
  for command,logname,budget in [(['vlib.exe','work'],'vlib.log',15),
      (['vlog.exe','-sv',*inp['files']],'compile.log',40),
      (['vsim.exe','-c','work.tb','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],'simulation.log',90)]:
   with (root/logname).open('x') as log:
    child=subprocess.Popen([str(q/command[0]),*command[1:]],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
    try:code=child.wait(timeout=budget)
    except subprocess.TimeoutExpired:stop_launched_tree(child);raise
   assert code==0,(logname,code)
  assert 'PASS COMPLETE_NEXT_STATE checks=90112 fault_rejected=1' in (root/'simulation.log').read_text()
  for n,h in inp['files'].items():assert sha(root/n)==h
  result=dict(status='COMPLETE_NEXT_STATE_90112_OFFICIAL_LUT_CHECKS_PASS',inputs_sha256=sha(root/'inputs.json'),
    checks=90112,fault_rejected=1,seconds=time.monotonic()-started,log_sha256=sha(root/'simulation.log'),hardware_access=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
 except Exception as exc:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(exc),seconds=time.monotonic()-started),indent=2));raise
if __name__=='__main__':main()
