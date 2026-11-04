"""Bounded pin tests for the new, fully mapped sensor configuration."""
import argparse,json,os,re,shutil,subprocess,time
from pathlib import Path
from run_units import ROOT,EF,sha
from windows_process_tree_safe import stop_launched_tree


def main():
 p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 prepared=a.prepared.resolve();root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp/frame-execution') and not root.exists()
 meta=json.loads((prepared/'inputs.json').read_text());board=Path(meta['board'])
 assert meta['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v')
 assert meta['audit_sha256']==sha(board/'sensor72-map-audit.json')
 for f,h in meta['files'].items():assert sha(f)==h
 root.mkdir();files={}
 for path in meta['files']:
  src=Path(path)
  if src.name.startswith('efx_'):assert sha(src)==sha(EF/'sim_models/verilog'/src.name)
  shutil.copy2(src,root/src.name);files[str(root/src.name)]=sha(root/src.name)
 inputs=dict(prepared=str(prepared),prepared_sha256=sha(prepared/'inputs.json'),files=files,runner_sha256=sha(__file__),hardware_access=False)
 (root/'inputs.json').write_text(json.dumps(inputs,indent=2))
 tool=Path('E:/quatra/questa_fse/win64');env=os.environ.copy();env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 def run(args,label,limit):
  start=time.monotonic();proc=subprocess.Popen(list(map(str,args)),cwd=root,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
  try:s,_=proc.communicate(timeout=limit)
  except subprocess.TimeoutExpired:
   (root/(label+'.timeout.json')).write_text(json.dumps(stop_launched_tree(proc),indent=2));raise
  (root/label).write_text(s,encoding='utf8')
  assert proc.returncode==0 and not re.search(r'\*\* (?:Error|Fatal)',s),(label,proc.returncode,s[-2000:])
  return s,dict(exit_code=proc.returncode,seconds=time.monotonic()-start,log_sha256=sha(root/label))
 run([tool/'vlib.exe','work'],'vlib.log',20)
 run([tool/'vlog.exe','-sv','-timescale','1ns/1ps','mapped_camera.v','tb.sv',*[k.lower()+'.v' for k in meta['kinds']]],'compile.log',60)
 (root/'run.do').write_text('onerror {quit -code 1}\nonbreak {quit -code 1}\nrun -all\nquit -code 0\n')
 cases=[]
 for mode in (0,10,4,6,8):
  s,r=run([tool/'vsim.exe','-c','work.tb_camera_checked_config',f'+MODE={mode}','-do','run.do'],f'mode{mode}.log',240)
  lines=[x for x in s.splitlines() if 'PASS CHECKED_CAMERA' in x];assert len(lines)==1,(mode,s[-2000:])
  cases.append(dict(mode=mode,pass_line=lines[0],**r));(root/'stages.json').write_text(json.dumps(cases,indent=2));print(lines[0],flush=True)
 for f,h in files.items():assert sha(f)==h
 result=dict(status='SENSOR72_MAPPED_I2C_PINS_PASS',cases=cases,inputs_sha256=sha(root/'inputs.json'),unmodified_official_models=True,hardware_access=False,board_release=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(result['status'],flush=True)


if __name__=='__main__':main()
