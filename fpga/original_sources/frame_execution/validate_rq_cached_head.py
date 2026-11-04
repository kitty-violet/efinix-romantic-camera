"""Official four-state models: independent FIFO oracle plus whole-RQ lockstep."""
import argparse,json,re,shutil,traceback
from pathlib import Path
from run_units import ROOT,EF,MS,run,sha
from rq_cached_head_board import BASE,REL,audit_delta
from validate_registered_ownership import rq_tb

def queue_module(text,name):
    lo=text.index(' wire [2:0] queue_bad,nonempty;');hi=text.index(' assign busy=',lo)
    return '''module NAME(input clk,reset,cancel,input [2:0] pushing,out_ready,
 input [265:0] payload,output [2:0] out_valid,out_bad,output [26:0] out_x,
 output [767:0] out_data,output reg error);
 reg [3:0] reserved[0:2],stored[0:2];
 wire active=!reset && !cancel && !error;
 wire [265:0] inverted_payload=~payload;
 wire [3:0] token_valid={|pushing,3'd0};
 wire [1:0] kinds[0:3];assign kinds[3]=pushing[2] ? 2'd2:pushing[1] ? 2'd1:2'd0;
 wire issue=active && |pushing;wire [1:0] chosen=kinds[3];
'''.replace('NAME',name)+text[lo:hi]+' always @(posedge clk)if(reset || cancel)error<=0;else if(|queue_bad)error<=1;\nendmodule\n'

QUEUE_TB='''`timescale 1ns/1ps
module tb_queue;
 reg clk=0;always #5 clk=~clk;
 reg reset=1,cancel=0;reg [2:0] pushing=0,out_ready=0;reg [265:0] payload=0;
 wire [2:0] out_valid,out_bad;wire [26:0] out_x;wire [767:0] out_data;wire error;
 wire [2:0] ref_valid,ref_bad;wire [26:0] ref_x;wire [767:0] ref_data;wire ref_error;
 queue_new dut(.*);
 queue_old reference(.clk(clk),.reset(reset),.cancel(cancel),.pushing(pushing),.out_ready(out_ready),.payload(payload),
  .out_valid(ref_valid),.out_bad(ref_bad),.out_x(ref_x),.out_data(ref_data),.error(ref_error));
 reg [265:0] expected[0:2][0:7];integer count[0:2];integer coverage[0:8][0:3];
 integer checks=0,cycles=0,seed=51,recoveries=0;reg checking=1;
 task compare;
  begin
   if({out_valid,error}!=={ref_valid,ref_error})$fatal(1,"QUEUE_CONTROL_DIFF");
   for(integer k=0;k<3;k=k+1)begin
    if(out_valid[k])begin
     if({out_bad[k],out_x[k*9+:9],out_data[k*256+:256]}!=={ref_bad[k],ref_x[k*9+:9],ref_data[k*256+:256]})$fatal(1,"QUEUE_DIFF");
     if(checking && (count[k]==0 || {out_bad[k],out_x[k*9+:9],out_data[k*256+:256]}!==expected[k][0]))$fatal(1,"QUEUE_ORACLE k=%0d n=%0d",k,count[k]);
     checks=checks+1;
    end
    if(checking && !reset && !cancel && !error && out_valid[k] !== (count[k]!=0))$fatal(1,"QUEUE_VALID");
   end
  end
 endtask
 always @(posedge clk)begin
  compare();cycles=cycles+1;
  for(integer k=0;k<3;k=k+1)begin
   if(reset || cancel)count[k]=0;
   else if(checking && !error)begin
    reg pop,push;pop=out_valid[k] && out_ready[k];push=pushing[k];
    coverage[count[k]][{push,pop}]=coverage[count[k]][{push,pop}]+1;
    if(pop)begin for(integer n=0;n<7;n=n+1)expected[k][n]=expected[k][n+1];count[k]=count[k]-1;end
    if(push)begin if(count[k]>=8)$fatal(1,"ORACLE_OVERFLOW");expected[k][count[k]]=payload;count[k]=count[k]+1;end
   end
  end
 end
 always @(negedge clk)begin #1;compare();end
 task tick(input [2:0] p,r);
  begin
   @(negedge clk);pushing=p;out_ready=r;
   for(integer j=0;j<9;j=j+1)payload[j*29+:29]=29'($random(seed));payload[265:261]=5'($random(seed));
  end
 endtask
 initial begin
  for(integer k=0;k<3;k=k+1)count[k]=0;
  for(integer n=0;n<9;n=n+1)for(integer a=0;a<4;a=a+1)coverage[n][a]=0;
  repeat(3)@(negedge clk);reset=0;
  // Every legal occupancy and push/pop combination, including full replacement.
  for(integer k=0;k<3;k=k+1)for(integer n=0;n<=8;n=n+1)begin
   cancel=1;tick(0,0);cancel=0;
   for(integer j=0;j<n;j=j+1)tick(3'b001<<k,0);
   tick(0,0);tick(0,0);
   if(n!=0)begin tick(3'b001<<k,3'b001<<k);tick(0,0);tick(0,3'b001<<k);tick(0,0);end
  end
  for(integer i=0;i<12000;i=i+1)begin
   integer k;reg [2:0] r,p;
   @(negedge clk);k=($random(seed)&32'h7fffffff)%3;r=3'($random(seed));p=0;
   if(count[k]<8 || r[k])p=3'b001<<k;
   pushing=p;out_ready=r;
   for(integer j=0;j<9;j=j+1)payload[j*29+:29]=29'($random(seed));
   payload[265:261]=5'($random(seed));
   // Four-state data must propagate identically, never become control.
   if(i%97==0)payload[103:80]=24'hx;
   cancel=i%431==430;reset=i%719==718;
  end
  tick(0,7);reset=0;cancel=0;repeat(10)tick(0,7);
  // Deliberate overflow must stop both versions and recover on reset.
  checking=0;tick(0,0);reset=1;tick(0,0);reset=0;
  repeat(9)tick(1,0);tick(0,0);tick(0,0);
  if(!error || !ref_error || out_valid!=0)$fatal(1,"OVERFLOW_NOT_STOPPED");
  reset=1;tick(0,0);reset=0;checking=1;recoveries=recoveries+1;
  tick(1,0);tick(0,7);repeat(8)tick(0,7);
  for(integer n=0;n<=8;n=n+1)begin
   if(coverage[n][0]==0)$fatal(1,"NO_IDLE n=%0d",n);
   if(n<8 && coverage[n][2]==0)$fatal(1,"NO_PUSH n=%0d",n);
   if(n>0 && (coverage[n][1]==0 || coverage[n][3]==0))$fatal(1,"NO_POP_PAIR n=%0d",n);
  end
  $display("PASS CACHED_HEAD_QUEUE cycles=%0d valid_compares=%0d all_legal_occupancy_actions=33 overflow_recovery=%0d",cycles,checks,recoveries);$finish;
 end
 initial begin #200000;$fatal(1,"TIMEOUT");end
endmodule
'''

def whole_tb():
    src,t=rq_tb()
    at=t.index(' reg [255:0] expected')
    t=t[:at]+''' wire [2:0] ref_in_ready,ref_out_valid,ref_out_bad;wire [767:0] ref_out_data;
 wire [26:0] ref_out_x;wire ref_busy,ref_error;integer differential_cycles=0;
 fixed_rq32_mixed_reference #(.REGISTER_SCALE_CLASS(1),.SHARED_FEEDBACK(TEST_SHARED)) reference(
 .clk(clk),.reset(reset),.cancel(cancel),.in_valid(in_valid),.in_ready(ref_in_ready),
 .mac_sums(mac_sums),.align_value(align_value),.post_value(post_value),.in_x(in_x),.feedback_value(feedback_value),
 .bias(bias),.multiplier(multiplier),.relu_enable(relu_enable),.mask(mask),.align_multiplier(align_multiplier),.post_multiplier(post_multiplier),
 .out_valid(ref_out_valid),.out_ready(out_ready),.out_data(ref_out_data),.out_x(ref_out_x),.out_bad(ref_out_bad),.busy(ref_busy),.error(ref_error));
 wire [287:0] signed_align;
 always @* feedback_value=dut.chosen==1 ? signed_align:post_value;
 for(genvar lane=0;lane<32;lane=lane+1)assign signed_align[lane*9+:9]={align_value[lane*8+7],align_value[lane*8+:8]};
 always @(negedge clk)begin
  #2;differential_cycles=differential_cycles+1;
  if({in_ready,out_valid,busy,error}!=={ref_in_ready,ref_out_valid,ref_busy,ref_error})$fatal(1,"RQ_PUBLIC_CYCLE_DIFF");
  for(integer k=0;k<3;k=k+1)if(out_valid[k] &&
   {out_bad[k],out_x[k*9+:9],out_data[k*256+:256]}!=={ref_out_bad[k],ref_out_x[k*9+:9],ref_out_data[k*256+:256]})$fatal(1,"RQ_PAYLOAD_DIFF");
 end
'''+t[at:]
    t=t.replace('module tb_fixed_split_rq;', 'module tb_fixed_split_rq;\n parameter TEST_SHARED=0;')
    t=t.replace('fixed_rq32_mixed #(.REGISTER_SCALE_CLASS(1)) dut',
                'fixed_rq32_mixed #(.REGISTER_SCALE_CLASS(1),.SHARED_FEEDBACK(TEST_SHARED)) dut')
    t=t.replace(' initial begin\n  for(integer k=0;', ' initial begin\n  if($value$plusargs("SEED=%d",seed))begin end\n  for(integer k=0;',1)
    t=t.replace('$finish;', '$display("PASS CACHED_HEAD_LOCKSTEP cycles=%0d shared=%0d",differential_cycles,TEST_SHARED);$finish;',1)
    return src,t

def main():
    p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    board=a.board.resolve();root=a.root.resolve();audit_delta(board)
    assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
    old=(BASE/REL).read_text();new=(board/REL).read_text();results=[]
    inputs=dict(board=str(board),board_inputs_sha256=sha(board/'inputs.json'),old_rq_sha256=sha(BASE/REL),new_rq_sha256=sha(board/REL),runner_sha256=sha(__file__),snapshots=[])
    try:
        for kind in ['queue','rq']:
            r=root/kind;r.mkdir()
            if kind=='queue':
                texts={'queues.sv':queue_module(old,'queue_old')+queue_module(new,'queue_new'),'tb_queue.sv':QUEUE_TB}
                top='tb_queue';models=['efx_srl8.v'];cases=[(0,51)]
            else:
                src,tb=whole_tb();inputs['integer_oracle_source']=str(src);inputs['integer_oracle_source_sha256']=sha(src)
                texts={'candidate.sv':new,'reference.sv':old.replace('fixed_rq32_mixed','fixed_rq32_mixed_reference').replace('fixed_rq_predecoded_pipeline','fixed_rq_predecoded_pipeline_reference'),'tb_fixed_split_rq.sv':tb}
                top='tb_fixed_split_rq';models=['efx_srl8.v','efx_dsp48.v'];cases=[(0,71),(1,391),(1,1027)]
            for name,body in texts.items():(r/name).write_text(body,encoding='utf8')
            for name in models:shutil.copy2(EF/'sim_models/verilog'/name,r/name)
            files=[r/n for n in [*texts,*models]]
            inputs['snapshots'] += [dict(path=str(f),sha256=sha(f)) for f in files]
            (root/'inputs.json').write_text(json.dumps(inputs,indent=2))
            run([MS/'vlib.exe','work'],r,'vlib.log',30)
            body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*files],r,'compile.log',60)
            assert 'Errors: 0, Warnings: 0' in body,body[-3000:]
            for shared,seed in cases:
                log=f'simulation-{shared}-{seed}.log'
                args=[MS/'vsim.exe','-c','work.'+top,*([f'-gTEST_SHARED={shared}'] if kind=='rq' else []),f'+SEED={seed}','-do','onerror {quit -code 1}; run -all; quit -code 0']
                body,sec=run(args,r,log,90)
                assert not re.search(r'(?m)^#?\s*(FAIL|\*\* (Error|Fatal))',body),body[-3000:]
                lines=re.findall(r'^# (PASS .*)$',body,re.M)
                assert len(lines)==(2 if kind=='rq' else 1) and 'Errors: 0, Warnings: 0' in body,body[-3000:]
                row=dict(kind=kind,shared_feedback=shared,seed=seed,seconds=sec,lines=lines,log_sha256=sha(r/log));results.append(row);print(json.dumps(row),flush=True)
        audit_delta(board)
        for row in inputs['snapshots']:assert sha(row['path'])==row['sha256']
        result=dict(status='RQ_CACHED_HEAD_QUEUE_NUMERIC_LOCKSTEP_PASS',inputs_sha256=sha(root/'inputs.json'),results=results,public_cycles_changed=False,hardware_access=False,board_release=False)
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    except Exception as exc:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc(),completed=results),indent=2));raise
if __name__=='__main__':main()
