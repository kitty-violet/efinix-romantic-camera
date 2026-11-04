"""Check the exact mapped cell with the official model; no timing exceptions."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,EF,MS,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--board',type=Path,required=True);a=p.parse_args()
r=a.root.resolve();base=a.board.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
net=base/'outflow/ti60f225_oob.map.v';body=net.read_text()
cell='LUT__127260_LUT__128392'
inst=re.search(r'EFX_COMB4 '+cell+r'\s*\((.*?)\);',body,re.S)
# The original netlist puts an attribute comment between ')' and ';'.
if not inst:inst=re.search(r'EFX_COMB4 '+cell+r'\s*\((.*?)\)\s*/\*.*?\*/\s*;',body,re.S)
assert inst and '.I3(1\'b0)' in inst[1] and '.I1(n60469)' in inst[1] and '.O(n60098)' in inst[1]
assert f'defparam {cell}.LUTMASK = 16\'h8850;' in body
model=EF/'sim_models/verilog/efx_comb4.v';shutil.copy2(model,r/model.name)
tb='''module tb;
reg i0,i1,i2;wire o,p;integer n,checks=0;
EFX_COMB4 #(.MODE("LOGIC"),.LUTMASK(16'h8850)) dut(.I0(i0),.I1(i1),.I2(i2),.I3(1'b0),.CI(1'b0),.O(o),.P(p),.CO(),.FCO());
initial begin
 for(n=0;n<8;n=n+1)begin
  {i2,i1,i0}=n;#1;
  if(o!==(i2 & ~i0) || p!==(i0 & i1))$fatal(1,"COMB4 truth mismatch");checks=checks+1;
 end
 for(n=0;n<4;n=n+1)begin
  {i2,i0}=n;i1=1'bx;#1;
  if(o!==(i2 & ~i0))$fatal(1,"COMB4 unused input X contaminated O");checks=checks+1;
 end
 $display("PASS COMB4_BUSY_ARC checks=%0d O_independent_of_I1_when_I3_zero=1 P_uses_I1=1",checks);$finish;
end
endmodule
'''
(r/'tb.sv').write_text(tb)
inputs=dict(map=str(net),map_sha256=sha(net),cell=cell,instance=inst[0],model_sha256=sha(model),tb_sha256=sha(r/'tb.sv'),runner_sha256=sha(__file__))
(r/'inputs.json').write_text(json.dumps(inputs,indent=2))
run([MS/'vlib.exe','work'],r,'vlib.log')
run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',r/'efx_comb4.v',r/'tb.sv'],r,'compile.log')
log,seconds=run([MS/'vsim.exe','-c','work.tb','-do','onerror {quit -code 1}; run -all; quit -code 0'],r,'simulation.log',timeout=30)
assert 'PASS COMB4_BUSY_ARC checks=12' in log and not re.search(r'\*\* (Error|Fatal)',log)
result=dict(status='OFFICIAL_COMB4_BOOLEAN_DEPENDENCY_PASS',checks=12,seconds=seconds,inputs_sha256=sha(r/'inputs.json'),log_sha256=sha(r/'simulation.log'),timing_exception_added=False,
scope='Selected reported I1-to-O arc is logically insensitive with I3=0; P still uses I1. Does not waive STA or prove hardware delay. Test RTL synthesis isolation instead.')
(r/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
