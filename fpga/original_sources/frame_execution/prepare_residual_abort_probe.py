"""Add active residual-state cancellation and tag-fault probes to numeric TB."""
import argparse
from pathlib import Path


def transform(body):
    body=body.replace('reg cfg_pw_fuse=0,','reg hold_skip_responses=0,inject_skip=0;integer residual_abort_checks=0;reg saw_skip_fault=0;\n reg cfg_pw_fuse=0,',1)
    needle=' .sv(sv&rsp_gate&{24{run_mode}}),.sr(csr),.sd(sd),.st(st^{24{inject ? 16\'h4000 : 16\'d0}}),.sw(sw),'
    assert body.count(needle)==1
    body=body.replace(needle,' .sv(sv&rsp_gate&{24{run_mode}}),.sr(csr),.sd(sd),.st(injected_tags),.sw(sw),')
    pos=body.index(' compute_column_cluster dut(')
    body=body[:pos]+''' wire [383:0] injected_tags;
 for(genvar pp=0;pp<24;pp=pp+1)begin:FAULT_TAG
  assign injected_tags[pp*16+:16]=st[pp*16+:16] ^ ((inject || (inject_skip && dut.skip_locked[pp])) ? 16'h4000:16'd0);
 end
 always @(posedge clk)if(inject_skip && dut.fuse_error)saw_skip_fault=1;
'''+body[pos:]
    needle='  if(run_mode && pause_writes!=0)'
    body=body.replace(needle,'  if(hold_skip_responses)rsp_gate=rsp_gate & ~dut.skip_locked;\n'+needle,1)
    needle='''    verify();blocks=blocks+1;
   end
  end'''
    assert body.count(needle)==1
    body=body.replace(needle,'''    verify();blocks=blocks+1;
   end
   if(caseid==9)begin
    for(integer probe=0;probe<5;probe=probe+1)begin
     initialize();run_mode=1;random_stall=1;hold_skip_responses=probe==1;
     start_group(0);waits=0;
     while(!(probe==0 || probe==1 || probe==4 ? (|dut.skip_fuse.pending):
             probe==2 ? (|dut.skip_fuse.have):(|dut.skip_fuse.qvalid)))begin
      @(negedge clk);waits=waits+1;if(waits>2000 || error)fail("residual probe state unreachable");
     end
     if(probe==1)repeat(20)@(negedge clk);
     if(probe==4)inject_skip=1;
     else begin cancel=1;repeat(2)@(negedge clk);cancel=0;end
     finish_group(1);inject_skip=0;hold_skip_responses=0;
     if(probe==4 && !saw_skip_fault)fail("residual tag fault was not detected by skip path");
     if(dut.skip_fuse.busy || dut.skip_locked!=0)fail("residual ownership survived cancellation");
     initialize();run_mode=1;start_group(0);finish_group(0);verify();
     residual_abort_checks=residual_abort_checks+1;
    end
   end
  end''',1)
    needle='  $display("COVERAGE blocks='
    body=body.replace(needle,'  if(residual_abort_checks!=5)fail("residual abort coverage");\n  $display("PASS RESIDUAL_ABORT_PROBE stages=%0d tag_fault=%b",residual_abort_checks,saw_skip_fault);\n'+needle,1)
    return body


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args()
    print(transform(a.base.read_text()),end='')
