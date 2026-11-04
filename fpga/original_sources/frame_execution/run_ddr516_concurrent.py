"""Run a hash-checked, already compiled real-DDR concurrency fixture."""
import argparse,json,os,re,subprocess,time
from pathlib import Path
from run_units import ROOT,sha

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp')
meta=json.loads((root/'inputs.json').read_text());prepared=json.loads((root/'result.json').read_text())
assert meta['concurrent'] and prepared['status']=='DDR516_MATRIX_COMPILE_PASS'
assert prepared['inputs_sha256']==sha(root/'inputs.json')
for row in meta['inputs']:assert sha(row['snapshot'])==row['sha256']
assert not (root/'concurrent_result.json').exists()
env=os.environ.copy();env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
command=['E:/quatra/questa_fse/win64/vsim.exe','-c','-voptargs=-O5','work.efx_ddr3_soft_controller_tb','-do','run.do']
start=time.monotonic();proc=subprocess.Popen(command,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace')
(root/'active_process.json').write_text(json.dumps(dict(pid=proc.pid,command=command,stage='concurrent_real_ddr',started=time.time())))
lines=[]
with (root/'concurrent_simulation.log').open('w',encoding='utf-8') as log:
 for line in proc.stdout:
  log.write(line);log.flush();lines.append(line)
  if re.search(r'CONCURRENT_|DDR516_CAL_DONE|DDR516_PROGRESS|\*\* (?:Fatal|Error)|\bERROR:',line):
   print(line.strip(),flush=True)
ret=proc.wait();body=''.join(lines)
errors=[line.strip() for line in lines if re.search(r'\*\* (?:Fatal|Error)|\bERROR:',line)]
passes=re.findall(r'^# CONCURRENT_PASS.*$',body,re.M)
passed=ret==0 and not errors and len(passes)==7 and 'DDR3 Controller PASS' in body
for row in meta['inputs']:assert sha(row['snapshot'])==row['sha256']
result=dict(status='DDR516_REAL_CONCURRENT_PASS' if passed else 'DDR516_REAL_CONCURRENT_FAIL',
 seconds=time.monotonic()-start,returncode=ret,errors=errors,passed_cases=passes,
 markers=[line.strip() for line in lines if 'CONCURRENT_' in line or 'DDR516_TRAFFIC' in line],
 inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'concurrent_simulation.log'),
 runner_sha256=sha(__file__),physical_pass=False,full_board=False,calibration_forced=False)
(root/'concurrent_result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if not passed:raise SystemExit(1)
