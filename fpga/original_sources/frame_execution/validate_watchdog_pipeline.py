"""Compare actual watchdog RTL to the old per-edge recurrence, including boundaries."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,MS,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
src=HERE/'variants/fixed_watchdog_pipeline/cnn_frame_datapath.sv';dst=r/src.name;shutil.copy2(src,dst)
tb=r/'tb_watchdog_pipeline.sv'
tb.write_text('''`timescale 1ns/1ps
module tb_watchdog_pipeline;
 reg clk=0,reset=1,work_active=0;reg [13:0] progress_events=0;
 always #5 clk=~clk;
 integer seed=2197,checks=0;reg [6:0] seen=0;
 for(genvar g=0;g<7;g=g+1)begin:G
  localparam integer C=g==0 ? 1:g==1 ? 2:g==2 ? 3:g==3 ? 7:g==4 ? 24:g==5 ? 256:1048576;
  localparam integer W=$clog2(C+1);
  reg [W-1:0] old_count=0;wire expired;wire old_expired=work_active && old_count>=W'(C-1);
  fixed_progress_watchdog #(.CYCLES(C)) dut(.*);
  always @(posedge clk)begin
   if(!reset)begin
    if(expired!==old_expired)$fatal(1,"WATCHDOG_EDGE C=%0d count=%0d old=%0d new=%0d",C,old_count,old_expired,expired);
    checks=checks+1;if(expired)seen[g]=1;
   end
   if(reset || !work_active || |progress_events)old_count<=0;
   else if(!old_expired)old_count<=old_count+1'b1;
  end
 end
 initial begin
  repeat(3)@(negedge clk);reset=0;
  // All 4^8 work/progress histories; resets between sequences.
  for(integer s=0;s<65536;s=s+1)begin
   reset=1;@(negedge clk);reset=0;
   for(integer k=0;k<8;k=k+1)begin
    work_active=(s>>(2*k))&1;progress_events=((s>>(2*k+1))&1)<<((s+k)%14);@(negedge clk);
   end
  end
  // Production-size saturation, restart, and progress at the deadline.
  work_active=1;progress_events=0;reset=1;@(negedge clk);reset=0;
  repeat(1048580)@(negedge clk);
  progress_events=1;@(negedge clk);progress_events=0;
  repeat(1048575)@(negedge clk);progress_events=14'h2000;@(negedge clk);
  work_active=0;@(negedge clk);progress_events=0;
  for(integer i=0;i<20000;i=i+1)begin
   work_active=($random(seed)&31)!=0;progress_events=($random(seed)&63)==0 ? 14'($random(seed)):0;
   reset=($random(seed)&255)==0;@(negedge clk);
  end
  if(seen!==7'h7f)$fatal(1,"WATCHDOG_COVERAGE seen=%h",seen);
  $display("PASS WATCHDOG_EXACT checks=%0d limits=7 histories=65536 production_expiry=1",checks);$finish;
 end
 initial begin #30000000;$fatal(1,"TIMEOUT");end
endmodule
''')
meta=dict(source=str(src),source_sha256=sha(src),snapshot_sha256=sha(dst),tb_sha256=sha(tb),runner_sha256=sha(__file__))
(r/'inputs.json').write_text(json.dumps(meta,indent=2))
run([MS/'vlib.exe','work'],r,'vlib.log');body,_=run([MS/'vlog.exe','-sv',dst,tb],r,'compile.log')
assert 'Errors: 0, Warnings: 0' in body
body,secs=run([MS/'vsim.exe','-c','work.tb_watchdog_pipeline','-do','onerror {quit -code 1}; run -all; quit -code 0'],r,'simulation.log',120)
lines=re.findall(r'^# (PASS WATCHDOG_EXACT.*)$',body,re.M);assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
assert sha(src)==sha(dst)==meta['source_sha256']
(r/'result.json').write_text(json.dumps(dict(status='WATCHDOG_CYCLE_EXACT_FOURSTATE_PASS',line=lines[0],seconds=secs,inputs_sha256=sha(r/'inputs.json'),log_sha256=sha(r/'simulation.log')),indent=2));print((r/'result.json').read_text())
