"""Four-state cache guard and unchanged independent arithmetic scoreboard."""
import argparse,json,os,re,shutil,traceback
from pathlib import Path
from run_units import ROOT,HERE,EF,run,sha
from accepted_tail_board import audit_delta
from control_command_repair import once

MS=Path('E:/quatra/questa_fse/win64')
GUARD=r'''`timescale 1ns/1ps
module tb_tail_guard;
 reg clk=0;always #5 clk=~clk;
 reg reset=1,cancel=0,idle=0,activate=0,begin_valid=0,load_valid=0,load_last=0,step=0;
 reg [2:0] begin_parts=0;reg [5:0] load_index=0;reg [63:0] load_data=0;
 wire begin_ready,load_ready,ready,error;wire [575:0] weights;wire [2:0] parts;wire [1:0] phase;
 fixed_ring_weights9 #(.SKEW_TERMS(1),.TAIL_SHIFT_048(1)) dut(.*);
 integer rejects=0,successes=0,no_bad_writes=0;
 for(genvar c=0;c<8;c=c+1)for(genvar b=0;b<8;b=b+1)begin:CHECK_WRITE
  always @(posedge clk)if(load_valid && load_ready && !dut.good && dut.TERM[8].CHANNEL[c].BIT[b].write!==1'b0)
   $fatal(1,"BAD_TAIL_WROTE_STORAGE");
 end
 task tick;begin @(posedge clk);#1;@(negedge clk);end endtask
 task clear(input integer rst);begin
  load_valid=0;begin_valid=0;activate=0;step=0;
  if(rst)reset=1;else cancel=1;tick();reset=0;cancel=0;#1;
  if(error || ready || !begin_ready)$fatal(1,"RECOVERY");
 end endtask
 task start(input integer n);begin
  begin_parts=3'(n);begin_valid=1;#1;if(!begin_ready)$fatal(1,"BEGIN");
  tick();begin_valid=0;
 end endtask
 task word(input integer idx,n,input [63:0] data);begin
  load_index=6'(idx);load_last=idx==9*n-1;load_data=data;load_valid=1;
  #1;if(!load_ready)$fatal(1,"LOAD");tick();load_valid=0;
 end endtask
 task good_group(input integer n);begin
  start(n);
  for(integer i=0;i<9*n;i=i+1)word(i,n,i%9==8 ? {8{8'd4}}:64'hff80fe017e83a55a);
  if(!ready || error)$fatal(1,"VALID_REJECTED");
  activate=1;tick();activate=0;successes=successes+1;
 end endtask
 initial begin
  repeat(3)@(negedge clk);reset=0;
  for(integer n=1;n<=4;n=n+1)for(integer c=0;c<8;c=c+1)for(integer v=0;v<256;v=v+1)
   if(v!=0 && v!=4 && v!=8)begin
    clear(v%2);good_group(n);start(n);
    // Invalid tail in the last phase, while the previous active group exists.
    for(integer i=0;i<9*n-1;i=i+1)begin
     step=i%2;word(i,n,i%9==8 ? {8{8'd8}}:64'hff80fe017e83a55a);
    end
    step=0;load_data={8{8'd4}};load_data[c*8+:8]=8'(v);
    word(9*n-1,n,load_data);
    if(!error || ready || load_ready || begin_ready || dut.loaded)$fatal(1,"INVALID_ACCEPTED n=%d c=%d v=%d",n,c,v);
    no_bad_writes=no_bad_writes+1;activate=1;tick();activate=0;
    if(!error || ready)$fatal(1,"BAD_ACTIVATION");rejects=rejects+1;
   end
  clear(1);good_group(4);clear(0);good_group(1);
  // Existing framing checks remain active, even with a supported value.
  clear(0);start(1);load_index=1;load_last=0;load_data=0;load_valid=1;tick();load_valid=0;
  if(!error)$fatal(1,"BAD_INDEX");
  clear(1);start(1);load_index=0;load_last=1;load_valid=1;tick();load_valid=0;
  if(!error)$fatal(1,"BAD_LAST");clear(0);good_group(3);
  if(rejects!=8096 || no_bad_writes!=8096 || successes!=8099)$fatal(1,"COVERAGE");
  $display("PASS TAIL_GUARD rejects=%0d recoveries=%0d no_bad_writes=%0d framing=2",rejects,successes,no_bad_writes);$finish;
 end
 initial begin #10000000;$fatal(1,"TIMEOUT");end
endmodule
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 board=a.board.resolve();root=a.root.resolve();audit_delta(board)
 assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
 os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 records=[];runs=[]
 for f in [board/'assembly/fixed_ring_weights9.sv',board/'assembly/fixed_window_dot8x32.sv',
           *[EF/'sim_models/verilog'/n for n in ('efx_srl8.v','efx_dsp24.v','efx_lut4.v')]]:
  d=root/f.name;shutil.copy2(f,d);records.append(dict(source=str(f),snapshot=str(d),sha256=sha(f)))
 cache=(HERE/'variants/fixed_history_test/tb_fixed_prefetch_weights.sv').read_text()
 cache=once(cache,'.SKEW_TERMS(1))','.SKEW_TERMS(1),.TAIL_SHIFT_048(1))')
 cache=once(cache,'  value=8\'((job*47+p*29+t*17+c*61)^(job<<3)^(p*c*7));',
  "  value=t==8 ? 8'(((job+p+c)%3)==0 ? 0:((job+p+c)%3)==1 ? 4:8):8'((job*47+p*29+t*17+c*61)^(job<<3)^(p*c*7));")
 arith=(HERE/'variants/fixed_shared_column_history_test/tb_fixed_window_dot8x32.sv').read_text()
 arith=once(arith,'.SHARED_COLUMN_ACTIVATION(1))','.SHARED_COLUMN_ACTIVATION(1),.TAIL_SHIFT_048(1))')
 anchor='    do begin out_ready=random_n(4)!=0;@(negedge clk);end while(!fired);'
 arith=once(arith,anchor,"    for(integer c=0;c<8;c=c+1)weight[(c*9+8)*8+:8]=8'((x+c)%3==0 ? 0:(x+c)%3==1 ? 4:8);\n"+anchor)
 benches={'tb_fixed_prefetch_weights':cache,'tb_tail_guard':GUARD,'tb_fixed_window_dot8x32':arith}
 for n,t in benches.items():(root/(n+'.sv')).write_text(t,encoding='utf8')
 manifest=dict(board_inputs_sha256=sha(board/'inputs.json'),inputs=records,runner_sha256=sha(__file__),
  testbenches={n:sha(root/(n+'.sv')) for n in benches},arithmetic_oracle='Existing independent signed sum scoreboard; only tail stimuli restricted to proven domain; other taps unrestricted',hardware_access=False)
 (root/'inputs.json').write_text(json.dumps(manifest,indent=2))
 try:
  run([MS/'vlib.exe','work'],root,'vlib.log')
  body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[x['snapshot'] for x in records],*[root/(n+'.sv') for n in benches]],root,'compile.log')
  assert 'Errors: 0, Warnings: 0' in body
  for top,budget in [('tb_fixed_prefetch_weights',90),('tb_tail_guard',120),('tb_fixed_window_dot8x32',240)]:
   body,seconds=run([MS/'vsim.exe','-c','work.'+top,'+SEED=71','-do','onerror {quit -code 1}; run -all; quit -code 0'],root,top+'.log',timeout=budget)
   assert 'Errors: 0, Warnings: 0' in body
   lines=re.findall(r'^# (PASS .*)$',body,re.M);assert lines
   runs.append(dict(top=top,seconds=seconds,lines=lines,log_sha256=sha(root/(top+'.log'))))
   print(json.dumps(runs[-1]),flush=True)
  audit_delta(board)
  for x in records:assert sha(x['source'])==sha(x['snapshot'])==x['sha256']
  result=dict(status='ACCEPTED_A_TAIL_FOUR_STATE_NUMERIC_GUARD_PASS',runs=runs,inputs_sha256=sha(root/'inputs.json'),public_cycles_changed=False,whole_board=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(result['status'],flush=True)
 except Exception as exc:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
