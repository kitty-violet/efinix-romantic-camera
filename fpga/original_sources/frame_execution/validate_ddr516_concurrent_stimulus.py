"""Check the new concurrent AXI stimulus, including negative controls."""
import argparse,json,shutil,subprocess
from pathlib import Path
from run_units import HERE,ROOT,MS,run,sha

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(exist_ok=False)
records=[]
for name in ('tb_ddr516_concurrent.sv','tb_ddr516_traffic_matrix_fixture.sv'):
 src=HERE/name;dst=root/name;shutil.copy2(src,dst)
 records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
(root/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2))
run([MS/'vlib.exe','work'],root,'vlib.log')
run([MS/'vlog.exe','-sv',*[row['snapshot'] for row in records]],root,'compile.log')
(root/'run.do').write_text('onerror {quit -code 1}\nonbreak {quit -code 1}\nrun -all\nquit -code 0\n')
cases=[]
for name,flags,expected in [('positive',[],'PASS MATRIX_STIMULUS_SEVEN_CASES writes=768 reads=896'),
 ('corrupt_read',['-gINJECT_READ_ERROR=1'],'CONCURRENT R mismatch'),
 ('stalled_write',['-gSTALL_W_CASE=4'],'CONCURRENT W timeout')]:
 with (root/(name+'.log')).open('w') as log:
  ret=subprocess.run(list(map(str,[MS/'vsim.exe','-c','work.tb_ddr516_traffic_matrix_fixture','-gCONCURRENT=1',*flags,'-do','run.do'])),
   cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=60).returncode
 body=(root/(name+'.log')).read_text();assert expected in body,(name,body[-2000:])
 if name=='positive':assert ret==0 and '** Fatal' not in body and body.count('# CONCURRENT_PASS ')==7
 else:assert 'PASS MATRIX_STIMULUS_SEVEN_CASES' not in body and '** Fatal' in body
 cases.append(dict(name=name,expected=expected,returncode=ret,log_sha256=sha(root/(name+'.log'))))
result=dict(status='CONCURRENT_STIMULUS_POSITIVE_AND_NEGATIVE_PASS',cases=cases,inputs_sha256=sha(root/'inputs.json'),actual_ddr=False)
(root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
