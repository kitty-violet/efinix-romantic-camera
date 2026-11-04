"""Exhaust all input bytes/weights by calling the actual scaler function."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,EF,MS,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);a=p.parse_args()
r=a.root.resolve();src=a.source.resolve()
assert r.is_relative_to(ROOT/'tmp') and not r.exists()
assert src.is_relative_to(HERE/'variants') and src.name=='rgb_bilinear_scaler_prototype.sv'
r.mkdir();base=ROOT/'tmp/frame-execution/connected-board-packed-display1/rtl'
files=[src,base/'resize_axis_up_2_to_3.sv',EF/'sim_models/verilog/efx_dpram10.v'];records=[]
for f in files:
 d=r/f.name;shutil.copy2(f,d);records.append(dict(source=str(f),snapshot=str(d),sha256=sha(f)))
tb=r/'tb_production_lerp.sv'
tb.write_text('''module tb_production_lerp;
 rgb_bilinear_scaler_prototype dut(.clk(1'b0),.rst_n(1'b0),.s_valid(1'b0),.s_ready(),
 .s_rgb(24'd0),.s_frame_id(16'd0),.s_sof(1'b0),.s_eol(1'b0),.m_valid(),.m_ready(1'b0),
 .m_rgb(),.m_frame_id(),.m_sof(),.m_eol(),.m_eof(),.protocol_error());
 integer expected,checks=0;
 initial begin
  for(integer a=0;a<256;a=a+1)for(integer b=0;b<256;b=b+1)for(integer w=0;w<8;w=w+1)begin
   case(w)
    0:expected=a;1:expected=(5*a+b+3)/6;3:expected=(a+b+1)/2;5:expected=(a+5*b+3)/6;default:expected=b;
   endcase
   if(dut.lerp(8'(a),8'(b),3'(w))!==8'(expected))$fatal(1,"LERP a=%0d b=%0d w=%0d",a,b,w);
   checks=checks+1;
  end
  $display("PASS PRODUCTION_LERP checks=%0d",checks);$finish;
 end
endmodule
''',encoding='utf-8')
(r/'inputs.json').write_text(json.dumps(dict(inputs=records,testbench_sha256=sha(tb),runner_sha256=sha(__file__)),indent=2))
run([MS/'vlib.exe','work'],r,'vlib.log')
body,_=run([MS/'vlog.exe','-sv',*[x['snapshot'] for x in records],tb],r,'compile.log')
assert 'Errors: 0, Warnings: 0' in body
body,secs=run([MS/'vsim.exe','-c','work.tb_production_lerp','-do','onerror {quit -code 1}; run -all; quit -code 0'],r,'simulation.log',90)
assert 'PASS PRODUCTION_LERP checks=524288' in body and 'Errors: 0, Warnings: 0' in body
for x in records:assert sha(x['source'])==sha(x['snapshot'])==x['sha256']
(r/'result.json').write_text(json.dumps(dict(status='EXHAUSTIVE_PRODUCTION_LERP_PASS',checks=524288,seconds=secs,inputs_sha256=sha(r/'inputs.json'),log_sha256=sha(r/'simulation.log')),indent=2))
print((r/'result.json').read_text())
