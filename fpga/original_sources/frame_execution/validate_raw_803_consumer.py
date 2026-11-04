"""Full-height 803-row recovery through Debayer/scaler/capture, narrow width.

Complements full-width short-height tests. Not a full1080p/board claim.
"""
from pathlib import Path
from run_units import sha

source=Path(__file__).with_name('validate_video_input_chain.py')
code=source.read_text()
needle=" (root/'tb.sv').write_text(tb)"
assert code.count(needle)==1
insert=''' assert a.kind=='recovery' and not a.wide and not a.isp
 assert tb.count('H=FULL ? 1080:24')==1
 tb=tb.replace('H=FULL ? 1080:24','H=1080')
 lo=tb.index('  good_frame(0);')
 hi=tb.index('\\n end\\n initial begin #(FULL',lo)
 tb=tb[:lo]+"""
  good_frame(0);
  camera_send(1,(W/4)*803,7);wait(aborts==1);wait(!abort_capture && dut.ingress.phase==0);
  repeat(100)@(negedge cam);good_frame(2);
  if(completions!=2 || failures!=1 || compared!=BYTES*2 || aw_count!=b_count || holding==0)
   $fatal(1,"803_CONSUMER_COVERAGE");
  $display("PASS RAW_803_DEBAYER_SCALER_CAPTURE width=%0d height=%0d good=%0d cancel=%0d bytes=%0d aw=%0d b=%0d held=%0d",W,H,completions,failures,compared,aw_count,b_count,holding);$finish;
"""+tb[hi:]
 tb=tb.replace('FULL ? 200000000:5000000','100000000')
'''
code=code.replace(needle,insert+needle)
code=code.replace('90 if a.wide else 60','90')
code=code.replace("runner_sha256=sha(__file__),scope=", "runner_sha256=sha(__file__),base_runner_sha256=sha(source),scope=")
exec(compile(code,str(source)+':803-full-height-consumer','exec'),globals())
