"""Exhaust the candidate's actual bias/range RTL against original signed sum."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,MS,sha,run
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--rtl',type=Path,required=True);a=p.parse_args()
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
src=a.rtl.resolve();body=src.read_text();shutil.copy2(src,r/src.name)
def extract(name):
 return body.split('// '+name+'_BEGIN',1)[1].split('// '+name+'_END',1)[0]
tb='''`timescale 1ns/1ps
module tb;
reg signed [23:0] low;reg signed [11:0] high_bias;
'''+extract('BIAS_RANGE_CLASS')+'''
wire [12:0] range_class=classify_bias_range(high_bias);
'''+extract('BIAS_RANGE_LANE')+'''
wire signed [12:0] reference_high=$signed({high_bias[11],high_bias})+$signed({{9{low[23]}},low[23:20]});
wire signed [32:0] reference_sum={reference_high,low[19:0]};
wire expected19=reference_sum[31:19]=={13{reference_sum[18]}};
wire expected25=reference_sum[31:25]=={7{reference_sum[24]}};
integer checks=0;
initial begin
 for(integer b=0;b<4096;b=b+1)for(integer c=0;c<16;c=c+1)for(integer t=0;t<4;t=t+1)begin
  high_bias=12'(b);low={4'(c),2'(t),18'h2aa55};#1;
  if({fits19,fits25}!=={expected19,expected25})
   $fatal(1,"BIAS_RANGE b=%h low=%h got=%b expected=%b",high_bias,low,{fits19,fits25},{expected19,expected25});
  checks=checks+1;
 end
 $display("PASS BIAS_RANGE_READY classes=%0d",checks);$finish;
end
endmodule
'''
(r/'tb.sv').write_text(tb)
(r/'inputs.json').write_text(json.dumps(dict(source=str(src),source_sha256=sha(src),extracted_sha256=sha(r/'tb.sv'),runner_sha256=sha(__file__)),indent=2))
run([MS/'vlib.exe','work'],r,'vlib.log')
body,_=run([MS/'vlog.exe','-sv','tb.sv'],r,'compile.log');assert 'Errors: 0, Warnings: 0' in body
body,seconds=run([MS/'vsim.exe','-c','work.tb','-do','onerror {quit -code 1}; run -all; quit -code 0'],r,'simulation.log',timeout=60)
assert 'PASS BIAS_RANGE_READY classes=262144' in body and not re.search(r'\*\* (Error|Fatal)',body),body[-2000:]
assert sha(src)==sha(r/src.name)
result=dict(status='BIAS_RANGE_READY_EXHAUSTIVE_PASS',classes=262144,seconds=seconds,source_sha256=sha(src),inputs_sha256=sha(r/'inputs.json'),log_sha256=sha(r/'simulation.log'),scope='All high12 bias, signed4 carry and two range bits; original signed33 sum reference; registered header contract checked by separate RQ and production gates')
(r/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
