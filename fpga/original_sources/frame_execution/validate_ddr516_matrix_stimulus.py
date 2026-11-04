"""Positive and fault-injection checks of the diagnostic traffic generator."""
import argparse,json,re,shutil,subprocess
from pathlib import Path
from run_units import ROOT,HERE,MS,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(exist_ok=False);records=[]
 for name in ('tb_ddr516_traffic_matrix.sv','tb_ddr516_traffic_matrix_fixture.sv'):
  src=HERE/name;dst=root/name;shutil.copy2(src,dst);records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
 (root/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2))
 def run(args,name):
  with (root/(name+'.log')).open('wb') as out:
   ret=subprocess.run(list(map(str,args)),cwd=root,stdout=out,stderr=subprocess.STDOUT,timeout=60).returncode
  return ret,(root/(name+'.log')).read_text(errors='replace')
 ret,body=run([MS/'vlib.exe','work'],'vlib');assert ret==0
 ret,body=run([MS/'vlog.exe','-sv',*[x['snapshot'] for x in records]],'compile');assert ret==0 and 'Errors: 0, Warnings: 0' in body
 cases=[]
 for name,options,expected in [('good',[],'PASS MATRIX_STIMULUS_SEVEN_CASES writes=640 reads=640'),
  ('corrupt',['-gINJECT_READ_ERROR=1'],'MATRIX R mismatch case=4 beat=0'),
  ('stall',['-gSTALL_W_CASE=3'],'MATRIX W timeout case=3 accepted=7')]:
  ret,body=run([MS/'vsim.exe','-c','work.tb_ddr516_traffic_matrix_fixture',*options,'-do','run -all; quit -code 0'],name)
  assert expected in body,(name,body[-2000:])
  if name=='good':assert ret==0 and '** Fatal' not in body
  else:assert '** Fatal' in body and 'PASS MATRIX_STIMULUS' not in body
  cases.append(dict(case=name,expected=expected,log_sha256=sha(root/(name+'.log'))))
 result=dict(status='MATRIX_STIMULUS_AND_FAILURE_DETECTION_PASS',cases=cases,inputs_sha256=sha(root/'inputs.json'),real_ddr=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
