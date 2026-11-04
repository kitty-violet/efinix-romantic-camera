"""Bounded official-DSP/SRL credit, cancellation and numerical fault gate."""
import argparse,json,os,re,shutil,subprocess,time
from pathlib import Path
from run_units import ROOT,HERE,EF,MS,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
v=HERE/'variants/fixed_shared_correction'
files=[v/'fixed_rq32_mixed.sv',HERE/'variants/fixed_shared_correction_test/tb_fixed_shared_correction.sv']+[EF/'sim_models/verilog'/n for n in ('efx_srl8.v','efx_dsp48.v')]
records=[]
for f in files:
 d=r/f.name;shutil.copy2(f,d);records.append(dict(source=str(f),snapshot=str(d),sha256=sha(f)))
(r/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2)+'\n')
run([MS/'vlib.exe','work'],r,'vlib.log');body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[x['snapshot'] for x in records]],r,'compile.log');assert 'Errors: 0, Warnings: 0' in body
scratch=r/'temp';scratch.mkdir();env=os.environ.copy();env.update(TEMP=str(scratch),TMP=str(scratch),TMPDIR=str(scratch))
t=time.monotonic();log=r/'simulation.log'
with log.open('w',encoding='utf-8') as f:
 proc=subprocess.run(list(map(str,[MS/'vsim.exe','-c','work.tb_fixed_shared_correction','-do','onerror {quit -code 1}; run -all; quit -code 0'])),cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=120)
body=log.read_text();assert proc.returncode==0 and not re.search(r'(?m)^#?\s*(FAIL|\*\* (Error|Fatal))',body),body[-2000:]
lines=re.findall(r'^# (PASS FIXED_SHARED_CORRECTION.*)$',body,re.M);assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
for x in records:assert sha(x['source'])==sha(x['snapshot'])==x['sha256']
result=dict(status='FIXED_SHARED_CORRECTION_PASS',line=lines[0],seconds=time.monotonic()-t,inputs_sha256=sha(r/'inputs.json'),log_sha256=sha(log))
(r/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
