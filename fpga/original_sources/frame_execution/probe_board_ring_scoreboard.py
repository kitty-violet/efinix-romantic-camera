"""Test the exact TMDS checker transformation, then measure isolated host work."""
import argparse,json,os,re,subprocess,time
from pathlib import Path
from run_units import ROOT,HERE,MS,sha
from board_ring_scoreboard import DECL,INIT,SHIFT,transform


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
 p.add_argument('--sim-bin',type=Path,default=MS);a=p.parse_args();sim=a.sim_bin.resolve()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
 sample=f'''module shift_monitor(input wire clk,input wire [72:0] value,output wire [72:0] delayed);
 {DECL}
 initial begin
 {INIT}
 end
 always @(posedge clk)begin
 {SHIFT}
 end
 assign delayed={{rgb_delay[58],de_delay[58],frame_delay[58],x_delay[58],y_delay[58]}};
 endmodule
 '''
 replacements={'{dut.hdmi_tx_rdata,dut.hdmi_tx_gdata,dut.hdmi_tx_bdata}':'value[72:49]',
  'dut.hdmi_tx_de && dut.hdmi_source_frame_active':'value[48]',
  'dut.hdmi_source_frame_id':'value[47:32]','dut.hdmi_timing_x':'value[31:16]','dut.hdmi_timing_y':'value[15:0]'}
 ring=transform(sample).replace('module shift_monitor','module ring_monitor',1)
 for old,new in replacements.items():sample=sample.replace(old,new);ring=ring.replace(old,new)
 tb='''`timescale 1ns/1ps
module tb;
 parameter MODE=0;parameter STEPS=100000;
 reg clk=0;always #5 clk=~clk;
 reg [72:0] value=0;wire [72:0] original,ring;
 reg [72:0] checksum=0;integer step,checks=0;
 if(MODE!=2)shift_monitor a(clk,value,original);
 if(MODE!=1)ring_monitor b(clk,value,ring);
 always @(posedge clk)begin
  #1;
  if(MODE==0 && original!==ring)$fatal(1,"TMDS ring delay mismatch step=%0d old=%h new=%h",step,original,ring);
  checksum=checksum^(MODE==1 ? original:ring);checks=checks+1;
 end
 initial begin
  for(step=0;step<STEPS;step=step+1)begin
   @(negedge clk);#1;
   value={24'(step*357),1'(step%3),16'(step/1125),16'(step%1920),16'(step%1080)};
   if(MODE==0)begin
    if(step%31==0)value[72:68]=5'bxz101;
    if(step%37==0)value[48]=1'bx;
    if(step%41==0)value[24]=1'bz;
   end
  end
  repeat(60)@(negedge clk);#1;
  $display("PASS BOARD_RING_CHECKER mode=%0d checks=%0d checksum=%h",MODE,checks,checksum);$finish;
 end
endmodule
'''
 (root/'old.sv').write_text(sample);(root/'ring.sv').write_text(ring);(root/'tb.sv').write_text(tb)
 board=ROOT/'tmp/frame-execution/board-full-raw-credit1/tb_board_startup.sv'
 frozen=board.read_text();changed=transform(frozen)
 (root/'tb_board_ring_candidate.sv').write_text(changed)
 manifest=dict(helper_sha256=sha(HERE/'board_ring_scoreboard.py'),runner_sha256=sha(__file__),
  frozen_tb_sha256=sha(board),candidate_tb_sha256=sha(root/'tb_board_ring_candidate.sv'),
  test_sha256=sha(root/'tb.sv'),old_sha256=sha(root/'old.sv'),ring_sha256=sha(root/'ring.sv'),
  simulator_bin=str(sim),running_board_modified=False,hardware_modified=False)
 (root/'inputs.json').write_text(json.dumps(manifest,indent=2))
 env=os.environ.copy();env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 def run(name,cmd):
  start=time.monotonic()
  with(root/f'{name}.log').open('wb')as log:ret=subprocess.run(list(map(str,cmd)),cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
  body=(root/f'{name}.log').read_text(errors='replace')
  assert ret.returncode==0 and '** Error' not in body and '** Fatal' not in body,name
  return time.monotonic()-start,body
 run('vlib',[sim/'vlib.exe','work']);run('compile',[sim/'vlog.exe','-sv','-timescale','1ns/1ps','old.sv','ring.sv','tb.sv'])
 (root/'run.do').write_text('onerror {quit -code 1}\nonbreak {quit -code 1}\nrun -all\nquit -code 0\n')
 results={}
 for mode,name in [(0,'equivalence'),(1,'shift_speed'),(2,'ring_speed')]:
  seconds,body=run(name,[sim/'vsim.exe','-c',*(['-voptargs=-O5'] if 'questa' in str(sim).lower() else []),'work.tb',f'-gMODE={mode}','-gSTEPS=10000' if mode==0 else '-gSTEPS=1000000','-do','run.do'])
  match=re.search(r'PASS BOARD_RING_CHECKER mode=(\d+) checks=(\d+) checksum=(\w+)',body);assert match and int(match[1])==mode
  results[name]=dict(seconds=seconds,checks=int(match[2]),checksum=match[3],log_sha256=sha(root/f'{name}.log'))
  print(name,round(seconds,3),flush=True)
 assert results['shift_speed']['checksum']==results['ring_speed']['checksum']
 result=dict(status='BOARD_TMDS_RING_CHECKER_EQUIVALENCE_PASS',measurements=results,
  isolated_host_speed_ratio=results['shift_speed']['seconds']/results['ring_speed']['seconds'],
  inputs_sha256=sha(root/'inputs.json'),latency_clock_edges=58,whole_board_speed_measured=False,
  running_board_modified=False,hardware_modified=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
