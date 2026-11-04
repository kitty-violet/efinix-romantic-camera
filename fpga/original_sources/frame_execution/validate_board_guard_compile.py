"""Compile updated measurement TB against frozen board sources, without running it."""
import argparse,json,os,re,shutil,subprocess
from pathlib import Path
from run_units import ROOT,HERE,sha
from prepare_board_frame_run import bench

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
 board=ROOT/'tmp/frame-execution/board-full-raw-credit1';meta=json.loads((board/'inputs.json').read_text())
 for row in meta['inputs']:assert sha(row['snapshot'])==row['sha256'],row['snapshot']
 (root/'rtl').mkdir();shutil.copy2(board/'rtl/ti60f225_oob_top.v',root/'rtl/ti60f225_oob_top.v')
 tb=bench(root,ROOT/'tmp/frame-execution/board-payload2',ddr516=True,
          camera_before_loader=True,warmup_lines=4,measurement_probes=True)
 assert tb.count('ddr516_measurement_guard ddr_guard(')==1
 (root/'tb_board_startup.sv').write_text(tb)
 sources=[Path(s) for s in meta['source_files'] if Path(s).name!='tb_board_startup.sv']
 sources += [HERE/'board_integration/ddr516_measurement_guard.sv',root/'tb_board_startup.sv']
 sim=Path('E:/quatra/questa_fse/win64');env=os.environ.copy();env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 commands=[('vlib',[sim/'vlib.exe','work']),('compile',[sim/'vlog.exe','-sv','-timescale','1ns/1ps',
 '+define+den2048Mb+sg125+x16',*['+incdir+'+p for p in meta['include_dirs']],*sources])]
 stages=[]
 for name,cmd in commands:
  with (root/(name+'.log')).open('wb') as out:r=subprocess.run(list(map(str,cmd)),cwd=root,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=120)
  body=(root/(name+'.log')).read_text(errors='replace')
  ok=r.returncode==0 and not re.search(r'\*\* (?:Error|Fatal)',body)
  stages.append(dict(stage=name,passed=bool(ok),returncode=r.returncode,log_sha256=sha(root/(name+'.log'))))
  if not ok:break
 passed=len(stages)==2 and all(s['passed'] for s in stages)
 result=dict(status='BOARD_GUARD_COMPILE_PASS' if passed else 'BOARD_GUARD_COMPILE_FAIL',stages=stages,
  frozen_inputs_sha256=sha(board/'inputs.json'),testbench_sha256=sha(root/'tb_board_startup.sv'),
  guard_sha256=sha(sources[-2]),simulation_started=False,elaboration_tested=False,full_board=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
 if not passed:raise SystemExit(1)
if __name__=='__main__':main()
