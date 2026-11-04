"""Run the exact extracted camera cone using unmodified vendor models.

The installed legacy simulator cannot start inside the current Windows sandbox
(WinError 623). The caller must run this offline test in a compatible context.
No USB, programming, network, or installation commands exist in this runner.
"""
import argparse, json, os, re, shutil, subprocess, time
from pathlib import Path
from run_units import ROOT, HERE, EF, MS, sha
from windows_process_tree_safe import stop_launched_tree


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True);p.add_argument('--questa',action='store_true')
 a=p.parse_args();root=a.root.resolve();prepared=a.prepared.resolve()
 assert root.is_relative_to(ROOT/'tmp/frame-execution') and not root.exists()
 source=json.loads((prepared/'inputs.json').read_text())
 assert source['runner_sha256']==sha(HERE/'test_video_sensor_rom_overlay.py')
 candidate=ROOT/'tmp/frame-execution/video-sensor-exposure71-candidate1/result.json'
 assert source['candidate_sha256']==sha(candidate)
 mapped=ROOT/'tmp/frame-execution/video-input-bit2/outflow/ti60f225_oob.map.v'
 assert source['base_map_sha256']==sha(mapped)
 for f,h in source['files'].items():assert sha(f)==h
 root.mkdir();files={}
 for name in ['mapped_camera.v','expected.hex','tb.sv',*[k.lower()+'.v' for k in source['proof']['kinds']]]:
  src=prepared/name
  if name.startswith('efx_'):assert sha(src)==sha(EF/'sim_models/verilog'/name)
  shutil.copy2(src,root/name);files[str(root/name)]=sha(root/name)
 inputs=dict(base_map_sha256=source['base_map_sha256'],candidate_sha256=sha(candidate),
  files=files,prepared_inputs=str(prepared/'inputs.json'),prepared_inputs_sha256=sha(prepared/'inputs.json'),
  extraction_runner_sha256=source['runner_sha256'],runner_sha256=sha(__file__),
  proof=source['proof'],actual_models='unmodified Efinity2026.1 official primitives',
  hardware_access=False)
 tool=Path('E:/quatra/questa_fse/win64') if a.questa else MS
 env=os.environ.copy()
 if a.questa:env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 inputs['simulator']=str(tool)
 (root/'inputs.json').write_text(json.dumps(inputs,indent=2))
 def run(args,label,limit):
  start=time.monotonic();proc=subprocess.Popen(list(map(str,args)),cwd=root,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
  try:s,_=proc.communicate(timeout=limit)
  except subprocess.TimeoutExpired:
   (root/(label+'.timeout.json')).write_text(json.dumps(stop_launched_tree(proc),indent=2));raise
  (root/label).write_text(s,encoding='utf8')
  assert proc.returncode==0 and not re.search(r'\*\* (?:Error|Fatal)',s),(label,proc.returncode,s[-2000:])
  return s,dict(exit_code=proc.returncode,seconds=time.monotonic()-start,log_sha256=sha(root/label))
 run([tool/'vlib.exe','work'],'vlib.log',20)
 run([tool/'vlog.exe','-sv','-timescale','1ns/1ps','mapped_camera.v','tb.sv',*[k.lower()+'.v' for k in source['proof']['kinds']]],'compile.log',60)
 (root/'run.do').write_text('onerror {quit -code 1}\nonbreak {quit -code 1}\nrun -all\nquit -code 0\n')
 cases=[]
 for mode in (0,4,6,8):
  s,result=run([tool/'vsim.exe','-c','work.tb_camera_checked_config',f'+MODE={mode}','-do','run.do'],f'mode{mode}.log',240)
  lines=[x for x in s.splitlines() if 'PASS CHECKED_CAMERA' in x];assert len(lines)==1,(mode,s[-1500:])
  cases.append(dict(mode=mode,pass_line=lines[0],**result));print(lines[0],flush=True)
 for f,h in files.items():assert sha(f)==h
 assert inputs['runner_sha256']==sha(__file__)
 result=dict(status='ACTUAL_MAPPED_EXPOSURE_CONTROLLER_PIN_TESTS_PASS',cases=cases,
  inputs_sha256=sha(root/'inputs.json'),unmodified_official_models=True,hardware_access=False,board_release=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(result['status'],flush=True)


if __name__=='__main__':main()
