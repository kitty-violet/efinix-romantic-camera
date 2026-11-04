"""Simulation-only inline expansion of the fixed public SRL8 bank history.
The measured synthesis source is never edited. Equivalence covers four-state
inputs, clock enables, all eight tap addresses, and the vendor INIT=0 behavior.
"""
import argparse,json,re,shutil,traceback
from pathlib import Path
from run_units import ROOT,HERE,EF,MS,sha,run
VECTOR="""        // Simulation projection of168 identical fixed-polarity EFX_SRL8 cells.
        reg [167:0] history_words[0:7];
        wire [167:0] incoming_prefix;
        wire history_ce;
        assign (weak0, weak1) history_ce=1'b1;
        assign history_ce=push_history;
        for(genvar l=0;l<8;l=l+1)begin:PACK
            assign incoming_prefix[l*21+:21]=selected_dots[l*32+:21];
        end
        initial begin
            for(integer d=0;d<8;d=d+1)history_words[d]=168'd0;
        end
        always @(posedge clk)if(history_ce)begin
            for(integer d=7;d>0;d=d-1)history_words[d]<=history_words[d-1];
            history_words[0]<=incoming_prefix;
        end
        // Official Q inversion followed by the RTL inversion converts Z to X.
        assign prior=~(~history_words[selected_age[2:0]]);
"""
def parts(body):
 start=body.index('        for(genvar l=0;l<8;l=l+1)begin:LANE')
 end=body.index('        always @(posedge clk)begin',start)
 original=body[start:end]
 expected="""        for(genvar l=0;l<8;l=l+1)begin:LANE
            for(genvar k=0;k<21;k=k+1)begin:BIT
                wire q;
                EFX_SRL8 history(.D(selected_dots[l*32+k]),.A(~selected_age[2:0]),
                    .CLK(clk),.CE(push_history),.Q(q),.Q7());
                assign prior[l*21+k]=~q;
            end
        end
"""
 assert original==expected
 return start,end,original
def project(body):
 start,end,_=parts(body)
 return body[:start]+VECTOR+body[end:]
def require_proof():
 root=ROOT/'tmp/frame-execution/bank-prefix-vector-proof3'
 proof=json.loads((root/'result.json').read_text())
 assert proof['status']=='SRL8_VECTOR_FOUR_STATE_EQUIVALENCE_PASS'
 assert proof['projector_sha256']==sha(__file__)
 assert proof['log_sha256']==sha(root/'simulation.log')
 assert proof['inputs_sha256']==sha(root/'inputs.json')
 meta=json.loads((root/'inputs.json').read_text())
 for row in meta['inputs']:assert sha(row['snapshot'])==row['sha256']
 return dict(path=str(root/'result.json'),sha256=sha(root/'result.json'),projector_sha256=sha(__file__))
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(exist_ok=False)
 source=ROOT/'tmp/frame-execution/connected-board-bank-prefix1/assembly/shared_rq_oc8_cluster.sv'
 body=source.read_text();_,_,original=parts(body)
 tb="""module tb_prefix_vector;
 reg clk=0;always #5 clk=~clk;
 reg push_history=0;reg [255:0] selected_dots=0;reg [3:0] selected_age=0;
 wire [167:0] data[0:1];
 for(genvar v=0;v<2;v=v+1)begin:VARIANT
  wire [167:0] prior;
  if(v==0)begin:OFFICIAL
__ORIGINAL__
  end else begin:VECTOR
__VECTOR__
  end
  assign data[v]=prior;
 end
 integer seed=4,checks=0,unknown=0;
 initial begin
  repeat(2)@(negedge clk);
  for(integer cycle=0;cycle<1024;cycle=cycle+1)begin
   for(integer l=0;l<8;l=l+1)selected_dots[l*32+:32]=$random(seed);
   if(cycle%7==0)begin selected_dots[17:3]='x;unknown=unknown+1;end
   if(cycle%11==0)begin selected_dots[132:122]='z;unknown=unknown+1;end
   push_history=cycle%13==0 ? 1'bx:cycle%17==0 ? 1'bz:cycle%3!=0;
   @(negedge clk);
   for(integer tap=0;tap<9;tap=tap+1)begin
    selected_age=tap==8 ? 4'b0xxx:4'(tap);#0.1;
    if(data[0]!==data[1])$fatal(1,"FAIL vector history cycle=%0d tap=%0d",cycle,tap);
    checks=checks+1;
   end
  end
  if(checks!=9216 || unknown<200)$fatal(1,"FAIL coverage");
  $display("PASS SRL8_VECTOR_FOUR_STATE checks=%0d unknown_inputs=%0d taps=8 unknown_tap=1 unknown_ce=1",checks,unknown);$finish;
 end
 initial begin #20000;$fatal(1,"FAIL timeout");end
endmodule
"""
 tb=tb.replace('__ORIGINAL__',original).replace('__VECTOR__',VECTOR)
 (root/'tb.sv').write_text(tb)
 official=EF/'sim_models/verilog/efx_srl8.v';shutil.copy2(official,root/official.name)
 meta=dict(inputs=[dict(snapshot=str(f),sha256=sha(f)) for f in (root/'tb.sv',root/official.name)],
  source=str(source),source_sha256=sha(source),projector_sha256=sha(__file__))
 (root/'inputs.json').write_text(json.dumps(meta,indent=2))
 run([MS/'vlib.exe','work'],root,'vlib.log')
 body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps','efx_srl8.v','tb.sv'],root,'compile.log')
 assert 'Errors: 0, Warnings: 0' in body
 body,seconds=run([MS/'vsim.exe','-c','work.tb_prefix_vector','-do','onerror {quit -code 1}; run -all; quit -code 0'],root,'simulation.log',timeout=60)
 assert 'PASS SRL8_VECTOR_FOUR_STATE checks=9216' in body and 'Errors: 0, Warnings: 0' in body
 result=dict(status='SRL8_VECTOR_FOUR_STATE_EQUIVALENCE_PASS',seconds=seconds,inputs_sha256=sha(root/'inputs.json'),
  log_sha256=sha(root/'simulation.log'),projector_sha256=sha(__file__),hardware_source_changed=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
