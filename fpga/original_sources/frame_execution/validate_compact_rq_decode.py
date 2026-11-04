"""Exhaust every high-bias/carry/range-bit class using the candidate RTL itself."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,MS,sha,run
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--rtl',type=Path,required=True);a=p.parse_args()
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
src=a.rtl.resolve();body=src.read_text();dst=r/src.name;shutil.copy2(src,dst)
start=body.index('  wire signed [6:0] high_low=');stop=body.index('  wire signed [7:0] a=',start)
decode=body[start:stop]
for name in ('fits19','fits25'):
 m=re.search(r'  wire '+name+r'=.*?;',body,re.S);assert m;decode+='\n'+m[0]
tb='''module tb;
reg signed [23:0] low;reg signed [11:0] high_bias;
'''+decode+'''
wire signed [12:0] reference_high=$signed({high_bias[11],high_bias})+$signed({{9{low[23]}},low[23:20]});
wire signed [32:0] reference_sum={reference_high,low[19:0]};
wire expected_overflow=reference_sum>33'sd2147483647 || reference_sum< -33'sd2147483648;
wire expected_fits19=reference_sum[31:19]=={13{reference_sum[18]}};
wire expected_fits25=reference_sum[31:25]=={7{reference_sum[24]}};
integer checks=0;
initial begin
 for(integer b=0;b<4096;b=b+1)for(integer c=0;c<16;c=c+1)for(integer t=0;t<4;t=t+1)begin
  high_bias=12'(b);low={4'(c),2'(t),18'h2aa55};#1;
  if({sum_overflow,sum_sign,fits19,fits25,high_low[4:0]}!==
   {expected_overflow,reference_sum[31],expected_fits19,expected_fits25,reference_high[4:0]})
   $fatal(1,"RQ_DECODE b=%h low=%h got=%b expected=%b",high_bias,low,
    {sum_overflow,sum_sign,fits19,fits25,high_low[4:0]},
    {expected_overflow,reference_sum[31],expected_fits19,expected_fits25,reference_high[4:0]});
  checks=checks+1;
 end
 $display("PASS RQ_DECODE classes=%0d",checks);$finish;
end
endmodule
'''
(r/'tb.sv').write_text(tb)
meta=dict(source=str(src),source_sha256=sha(src),extracted_sha256=sha(r/'tb.sv'),runner_sha256=sha(__file__))
(r/'inputs.json').write_text(json.dumps(meta,indent=2))
run([MS/'vlib.exe','work'],r,'vlib.log')
body,_=run([MS/'vlog.exe','-sv','tb.sv'],r,'compile.log');assert 'Errors: 0, Warnings: 0' in body
body,seconds=run([MS/'vsim.exe','-c','work.tb','-do','onerror {quit -code 1}; run -all; quit -code 0'],r,'simulation.log',timeout=60)
assert 'PASS RQ_DECODE classes=262144' in body and not re.search(r'\*\* (Error|Fatal)',body),body[-2000:]
assert sha(src)==sha(dst)==meta['source_sha256']
result=dict(status='RQ_DECODE_EXHAUSTIVE_CLASSES_PASS',classes=262144,seconds=seconds,
 source_sha256=sha(src),inputs_sha256=sha(r/'inputs.json'),log_sha256=sha(r/'simulation.log'),
 scope='All 12 high-bias bits, four signed carry bits, two range bits; lower18 bits bypass unchanged')
(r/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
