"""Official four-state arithmetic gate, no board/supply/performance claim."""
import argparse,concurrent.futures,json,os,re,shutil,subprocess,time
from pathlib import Path
from run_units import ROOT,HERE,EF,MS,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--reuse-clean',type=Path);p.add_argument('--testbench',type=Path);a=p.parse_args()
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
v=HERE/'variants/fixed_window_dot8x32';records=[]
arithmetic=Path(os.environ['FPGA_FIXED_ARITHMETIC']).resolve() if os.environ.get('FPGA_FIXED_ARITHMETIC') else v/'fixed_window_dot8x32.sv'
tb=a.testbench.resolve() if a.testbench else v/'tb_fixed_window_dot8x32.sv'
assert tb.is_relative_to(HERE/'variants') and tb.name=='tb_fixed_window_dot8x32.sv'
for f in [arithmetic,tb,EF/'sim_models/verilog/efx_dsp24.v',EF/'sim_models/verilog/efx_srl8.v']:
 dst=r/f.name;shutil.copy2(f,dst);records.append(dict(source=str(f),snapshot=str(dst),sha256=sha(f)))
(r/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2)+'\n')
run([MS/'vlib.exe','work'],r,'vlib.log');body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[x['snapshot'] for x in records]],r,'compile.log');assert 'Errors: 0, Warnings: 0' in body
def simulate(seed):
 # The observed eight-job prefix took120s in the vendor DSP model.
 # Twelve identical-coverage jobs get240s, each seed runs independently.
 logfile=r/f'simulation_{seed}.log';reused=False
 if a.reuse_clean:
  prev=a.reuse_clean.resolve();pi=json.loads((prev/'inputs.json').read_text())
  assert {Path(x['snapshot']).name:x['sha256'] for x in pi['inputs']}=={Path(x['snapshot']).name:x['sha256'] for x in records}
  for x in pi['inputs']:assert sha(x['snapshot'])==x['sha256']
  old=prev/logfile.name;body=old.read_text()
  if not re.search(r'(?m)^#?\s*(FAIL|\*\* (Error|Fatal))',body) and 'Errors: 0, Warnings: 0' in body and re.search(r'PASS FIXED_WINDOW_DOT8X32 seed='+str(seed)+r'\b',body):
   shutil.copy2(old,logfile);seconds=None;reused=True
 if not reused:
  scratch=r/f'seed_{seed}';scratch.mkdir();env=os.environ.copy();env.update(TEMP=str(scratch),TMP=str(scratch),TMPDIR=str(scratch))
  start=time.monotonic()
  with logfile.open('w',encoding='utf-8') as log:
   proc=subprocess.run(list(map(str,[MS/'vsim.exe','-c','-l',scratch/'transcript','-wlf',scratch/'wave.wlf','work.tb_fixed_window_dot8x32',f'+SEED={seed}','-do','onerror {quit -code 1}; run -all; quit -code 0'])),cwd=r,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
  seconds=time.monotonic()-start;body=logfile.read_text()
  assert proc.returncode==0 and not re.search(r'(?m)^#?\s*(FAIL|\*\* (Error|Fatal))',body),body[-2000:]
 lines=re.findall(r'^# (PASS FIXED_WINDOW_DOT8X32.*)$',body,re.M);assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
 result=dict(seed=seed,line=lines[0],seconds=seconds,reused_clean_log=str(a.reuse_clean.resolve()/logfile.name) if reused else None,log_sha256=sha(logfile));print(result,flush=True);return result
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(simulate,(71,391,1027)))
for x in records:assert sha(x['source'])==sha(x['snapshot'])==x['sha256']
(r/'result.json').write_text(json.dumps(dict(status='FIXED_WINDOW_DOT8X32_OFFICIAL_ARITHMETIC_PASS',runs=results,inputs_sha256=sha(r/'inputs.json'),real_supply=False,whole_cnn=False,physical=False),indent=2)+'\n')

