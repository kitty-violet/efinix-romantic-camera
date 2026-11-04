"""Real six-bank SRAM and finite64-bit weight-loader integration gate."""
import argparse,json,os,re,shutil,subprocess,time
from pathlib import Path
from run_units import ROOT,HERE,EF,MS,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
v=HERE/'variants/fixed_window_dot8x32';files=[v/n for n in ('fixed_window_dot8x32.sv','fixed_matrix_feed6.sv','fixed_ring_weights9.sv','fixed_partial32.sv','tb_fixed_matrix_feed6.sv')]
files+=[HERE/'variants/row_pair_arena/row_pair_arena6.sv']+[EF/'sim_models/verilog'/n for n in ('efx_dsp24.v','efx_srl8.v','efx_dpram10.v')]
records=[]
for f in files:
 dst=r/f.name;shutil.copy2(f,dst);records.append(dict(source=str(f),snapshot=str(dst),sha256=sha(f)))
(r/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2)+'\n')
run([MS/'vlib.exe','work'],r,'vlib.log');body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[x['snapshot'] for x in records]],r,'compile.log');assert 'Errors: 0, Warnings: 0' in body
scratch=r/'temp';scratch.mkdir();env=os.environ.copy();env.update(TEMP=str(scratch),TMP=str(scratch),TMPDIR=str(scratch))
start=time.monotonic();log=r/'simulation.log'
with log.open('w',encoding='utf-8') as f:proc=subprocess.run(list(map(str,[MS/'vsim.exe','-c','work.tb_fixed_matrix_feed6','+SEED=71','-do','onerror {quit -code 1}; run -all; quit -code 0'])),cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=240)
body=log.read_text();assert proc.returncode==0 and not re.search(r'(?m)^#?\s*(FAIL|\*\* (Error|Fatal))',body),body[-2500:]
lines=re.findall(r'^# (PASS FIXED_MATRIX_REAL_RAM.*)$',body,re.M);assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
for x in records:assert sha(x['source'])==sha(x['snapshot'])==x['sha256']
result=dict(status='FIXED_MATRIX_REAL_RAM_AND_WEIGHT_PORT_PASS',line=lines[0],jobs=re.findall(r'^# (MATRIX_RAM_JOB.*)$',body,re.M),seconds=time.monotonic()-start,inputs_sha256=sha(r/'inputs.json'),log_sha256=sha(log),full_cnn=False,rq=False,production_placement=False)
(r/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

