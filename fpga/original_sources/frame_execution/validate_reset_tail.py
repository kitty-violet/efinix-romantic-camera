"""Four-state, multi-clock reset-tail equivalence from exact frozen board RTL."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,MS,sha,run

p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
a=p.parse_args();base=a.base.resolve();r=a.root.resolve()
assert base.is_relative_to(ROOT/'tmp') and r.is_relative_to(ROOT/'tmp') and not r.exists()
original=base/'assembly/cnn_video_pipeline.sv'
candidate=HERE/'variants/reset_tail/cnn_video_pipeline.sv'
before=original.read_text();after=before
for domain in ('process','display'):
 old=f'wire {domain}_rst_n = &{domain}_reset_pipe;'
 new=f'wire {domain}_rst_n = {domain}_reset_pipe[3];'
 assert after.count(old)==1;after=after.replace(old,new)
assert candidate.read_text()==after,'Reject any unrelated reset or pipeline change'
r.mkdir();shutil.copy2(candidate,r/candidate.name)
def block(text):
 start=text.index('    reg [3:0] process_reset_pipe;')
 stop=text.index('    wire down_valid;',start)
 return text[start:stop]
duts=''
for name,body in [('original',before),('candidate',after)]:
 s=block(body).replace('wire process_rst_n =','assign process_rst_n =').replace('wire display_rst_n =','assign display_rst_n =')
 duts+='module reset_'+name+'(input wire reset,process_clk,display_clk,output wire process_rst_n,display_rst_n);\n'+s+'endmodule\n'
(r/'reset_views.sv').write_text(duts)
tb='''`timescale 1ns/1ps
module reset_case #(parameter real P=3.3335,D=6.734,PHASE=0.0)(output reg done=0);
 reg reset=1,process_clk=0,display_clk=0;reg process_running=1,display_running=1;
 wire op,od,cp,cd;
 reset_original old_dut(.process_rst_n(op),.display_rst_n(od),.*);
 reset_candidate new_dut(.process_rst_n(cp),.display_rst_n(cd),.*);
 initial forever begin #(P);if(process_running)process_clk=~process_clk;end
 initial begin #(PHASE);forever begin #(D);if(display_running)display_clk=~display_clk;end end
 integer checks=0,assertions=0,process_edges=0,display_edges=0,process_releases=0,display_releases=0;
 reg [4:0] states_p=0,states_d=0;
 task check;begin
  if({op,od}!=={cp,cd})$fatal(1,"RESET_TAIL_MISMATCH P=%f D=%f",P,D);
  if(reset && {cp,cd}!==2'b00)$fatal(1,"RESET_ASSERT_NOT_IMMEDIATE");
  if(!reset && (cp!==(process_edges>=4) || cd!==(display_edges>=4)))$fatal(1,"RESET_RELEASE_NOT_FOUR_EDGES");
  checks=checks+1;
 end endtask
 always @(posedge reset)begin
  process_edges=0;display_edges=0;assertions=assertions+1;#0.002;check();
 end
 always @(posedge process_clk)begin
  if(reset)process_edges=0;else if(process_edges<5)process_edges=process_edges+1;
  #0.002;check();
  if(!reset)begin
   if(process_edges==4)process_releases=process_releases+1;
   if(process_edges<=4)states_p[process_edges]=1;
  end else states_p[0]=1;
 end
 always @(posedge display_clk)begin
  if(reset)display_edges=0;else if(display_edges<5)display_edges=display_edges+1;
  #0.003;check();
  if(!reset)begin
   if(display_edges==4)display_releases=display_releases+1;
   if(display_edges<=4)states_d[display_edges]=1;
  end else states_d[0]=1;
 end
 initial begin
  #25.123;reset=0;
  for(integer i=0;i<160;i=i+1)begin
   #(31.117+(i%17)*7.031);
   reset=1;
   // Repeated assertion during partial release, sub-period pulses, and
   // stopped clocks: only actual local edges may release the output.
   process_running=i%9!=0;display_running=i%7!=0;
   #(0.127+(i%11)*1.113);reset=0;
   #70.131;process_running=1;display_running=1;
   if(i%5==0)begin #0.127;reset=1;#0.233;reset=0;end
  end
  #200;
  if(assertions<180 || process_releases<150 || display_releases<150 || states_p!==5'b11111 || states_d!==5'b11111)
   $fatal(1,"RESET_TAIL_COVERAGE %d %d %d %b %b",assertions,process_releases,display_releases,states_p,states_d);
  $display("PASS RESET_CASE P=%f D=%f phase=%f checks=%0d assertions=%0d releases=%0d/%0d",P,D,PHASE,checks,assertions,process_releases,display_releases);done=1;
 end
endmodule
module tb_reset_tail;
 wire [3:0] done;
 reset_case #(.P(3.3335),.D(6.734),.PHASE(0.411)) c0(done[0]);
 reset_case #(.P(2.5),.D(6.734),.PHASE(1.817)) c1(done[1]);
 reset_case #(.P(6.734),.D(3.3335),.PHASE(0.0)) c2(done[2]);
 reset_case #(.P(3.3335),.D(3.3335),.PHASE(3.331)) c3(done[3]);
 initial begin wait(&done);$display("PASS RESET_TAIL_CYCLE_EQUIVALENCE cases=4");$finish;end
 initial begin #100000;$fatal(1,"RESET_TAIL_TIMEOUT");end
endmodule
'''
(r/'tb_reset_tail.sv').write_text(tb)
inputs=dict(base_source=str(original),base_sha256=sha(original),candidate_source=str(candidate),candidate_sha256=sha(candidate),
 reset_views_sha256=sha(r/'reset_views.sv'),testbench_sha256=sha(r/'tb_reset_tail.sv'),runner_sha256=sha(__file__),delta='Only two four-input reductions replaced with last shift stage; no clock/latency/reset change')
(r/'inputs.json').write_text(json.dumps(inputs,indent=2)+'\n')
run([MS/'vlib.exe','work'],r,'vlib.log')
body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',r/'reset_views.sv',r/'tb_reset_tail.sv'],r,'compile.log');assert 'Errors: 0, Warnings: 0' in body
body,secs=run([MS/'vsim.exe','-c','work.tb_reset_tail','-do','onerror {quit -code 1}; run -all; quit -code 0'],r,'simulation.log',timeout=60)
assert 'PASS RESET_TAIL_CYCLE_EQUIVALENCE cases=4' in body and 'Errors: 0, Warnings: 0' in body
lines=re.findall(r'^# (PASS RESET_CASE.*)$',body,re.M);assert len(lines)==4
assert sha(original)==inputs['base_sha256'] and sha(candidate)==sha(r/candidate.name)==inputs['candidate_sha256']
result=dict(status='RESET_TAIL_CYCLE_EQUIVALENCE_PASS',cases=lines,seconds=secs,inputs_sha256=sha(r/'inputs.json'),log_sha256=sha(r/'simulation.log'),candidate_sha256=sha(candidate),base_sha256=sha(original),analog_metastability_proven=False,physical_timing_pass=False)
(r/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
