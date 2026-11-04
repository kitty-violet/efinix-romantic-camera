"""Bounded fixed startup-controller unit test; not a JTAG/CPU replacement gate."""
import argparse,json,os,re,shutil,subprocess,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,MS,sha,run
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--variant',type=Path,default=HERE/'variants/fixed_boot_sequencer');a=p.parse_args()
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
v=a.variant.resolve();assert v.is_relative_to(HERE/'variants');records=[]
for name in ('cnn_boot_sequencer.sv','tb_cnn_boot_sequencer.sv'):
 f=v/name
 if not f.exists():f=HERE/'variants/fixed_boot_sequencer'/name
 d=r/name;shutil.copy2(f,d);records.append(dict(source=str(f),snapshot=str(d),sha256=sha(f)))
(r/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2))
try:
 run([MS/'vlib.exe','work'],r,'vlib.log',timeout=30)
 body,_=run([MS/'vlog.exe','-sv',*[x['snapshot'] for x in records]],r,'compile.log',timeout=30)
 assert 'Errors: 0, Warnings: 0' in body
 scratch=r/'temp';scratch.mkdir();env=os.environ.copy();env.update(TEMP=str(scratch),TMP=str(scratch))
 t=time.monotonic()
 with (r/'simulation.log').open('w') as log:
  proc=subprocess.run([str(MS/'vsim.exe'),'-c','work.tb_cnn_boot_sequencer','-do','onerror {quit -code 1}; run -all; quit -code 0'],cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=90)
 body=(r/'simulation.log').read_text();assert proc.returncode==0 and not re.search(r'(?m)^#?\s*(?:FAIL|\*\* (?:Error|Fatal))',body)
 lines=re.findall(r'^# (PASS FIXED_BOOT_SEQUENCER.*)$',body,re.M);assert len(lines)==1
 for x in records:assert sha(x['source'])==sha(x['snapshot'])==x['sha256']
 result=dict(status='FIXED_BOOT_SEQUENCER_UNIT_PASS',line=lines[0],seconds=time.monotonic()-t,inputs_sha256=sha(r/'inputs.json'),log_sha256=sha(r/'simulation.log'),bus_model='bounded randomized posted-write/read-response behavioral bus; no JTAG or APB/DDR RTL',board_integrated=False)
 (r/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
except Exception as e:
 (r/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
