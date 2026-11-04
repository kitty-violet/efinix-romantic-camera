"""Prove actual packed feedback pin roles with the installed four-state model."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,EF,MS,run,sha
p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
b=a.board.resolve();r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
net=b/'outflow/ti60f225_oob.map.v';s=net.read_text()
cells=[]
for m in re.finditer(r'EFX_COMB4\s+(\S+)\s*\((.*?)\)\s*/\*.*?\*/\s*;',s,re.S):
 if '/feedback_selector' in m[1]:cells.append(m)
assert len(cells)==1
m=cells[0];name=m[1];pins=dict(re.findall(r'\.(I[0-3]|O|P)\(([^)]*)\)',m[2]))
assert pins['I0'].strip().endswith('/main_scaled')
assert pins['I1'].strip().endswith('/skip/skip_held')
assert pins['I2'].strip().endswith('/ir [2]') and pins['I3']=="1'b0"
assert pins['O'].strip().endswith('/skip/feedback_select')
mask=re.search(r'defparam '+re.escape(name)+r"\s*\.LUTMASK = 16'h([0-9a-fA-F]+);",s)[1]
assert int(mask,16)==0x445a
model=EF/'sim_models/verilog/efx_comb4.v';shutil.copy2(model,r/model.name)
tb="""module tb;
reg a,b,c;wire o,p;integer checks=0;
EFX_COMB4 #(.MODE("LOGIC"),.LUTMASK(16'h445a)) dut(.I0(a),.I1(b),.I2(c),.I3(1'b0),.CI(1'b0),.O(o),.P(p),.CO(),.FCO());
function value(input integer n);case(n)0:value=0;1:value=1;default:value=1'bx;endcase endfunction
initial begin
for(integer i=0;i<3;i=i+1)for(integer j=0;j<3;j=j+1)for(integer k=0;k<3;k=k+1)begin
a=value(i);b=value(j);c=value(k);#1;
if(o !== (a^c) || p !== (b && !a))$fatal(1,"PIN_ROLE_MISMATCH a=%b b=%b c=%b o=%b p=%b",a,b,c,o,p);
checks=checks+1;
end
$display("PASS MAPPED_FEEDBACK_PIN_ROLES checks=%0d",checks);$finish;
end
endmodule
"""
(r/'tb.sv').write_text(tb);shutil.copy2(__file__,r/'runner.py')
run([MS/'vlib.exe','work'],r,'vlib.log')
body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps','efx_comb4.v','tb.sv'],r,'compile.log');assert 'Errors: 0, Warnings: 0' in body
body,sec=run([MS/'vsim.exe','-c','work.tb','-do','onerror {quit -code 1}; run -all; quit -code 0'],r,'simulation.log',timeout=60)
assert 'PASS MAPPED_FEEDBACK_PIN_ROLES checks=27' in body and not re.search(r'\*\* (Error|Fatal)',body)
data=dict(status='MAPPED_FEEDBACK_PIN_ROLE_FOURSTATE_PASS_NOT_STA',map_sha256=sha(net),cell=name,pins=pins,mask=mask,model_sha256=sha(model),log_sha256=sha(r/'simulation.log'),seconds=sec,
 actual_primitive='COMB4_LOGIC',note='Explicit LUT was still packed, but ready is now I2 and the independent forward condition is P, whose physical/function inputs are I0/I1 only. Actual routed STA remains mandatory.',timing_exceptions_added=False)
(r/'result.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data))
