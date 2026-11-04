"""Short pin-level checks for the ROM-backed verification prototype.

Uses unmodified vendor primitives, but is deliberately NOT a fresh map/P&R
release check. No USB or board access.
"""
import argparse,json,os,re,shutil,subprocess,time
from pathlib import Path
from run_units import ROOT,EF,sha
from windows_process_tree_safe import stop_launched_tree

def main():
 p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--modes',default='0,10,8');a=p.parse_args()
 prepared=a.prepared.resolve();root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp/frame-execution') and not root.exists()
 meta=json.loads((prepared/'inputs.json').read_text());assert meta['status']=='SENSOR_ROM_VERIFY_CANDIDATE_NOT_SYNTHESIZED'
 assert meta['new_flip_flops']==0 and meta['new_clocks']==0
 for f,h in meta['files'].items():assert sha(f)==h,f
 assert sha(Path(__file__).with_name('prepare_video_sensor_rom_verify.py'))==meta['script_sha256']
 root.mkdir();files={}
 for f in meta['files']:
  src=Path(f)
  if src.name=='native_source.v':continue
  if src.name.startswith('efx_'):assert sha(src)==sha(EF/'sim_models/verilog'/src.name)
  shutil.copy2(src,root/src.name);files[str(root/src.name)]=sha(root/src.name)
 modes=[int(v) for v in a.modes.split(',')];assert modes and all(v in (0,4,6,8,10) for v in modes)
 inputs=dict(prepared=str(prepared),prepared_sha256=sha(prepared/'inputs.json'),files=files,runner_sha256=sha(__file__),modes=modes,hardware_access=False)
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
 for mode in modes:
  s,r=run([tool/'vsim.exe','-c','work.tb_camera_checked_config',f'+MODE={mode}','-do','run.do'],f'mode{mode}.log',240)
  lines=[x for x in s.splitlines() if 'PASS CHECKED_CAMERA' in x];assert len(lines)==1,(mode,s[-2000:])
  cases.append(dict(mode=mode,pass_line=lines[0],**r));(root/'stages.json').write_text(json.dumps(cases,indent=2));print(lines[0],flush=True)
 for f,h in files.items():assert sha(f)==h
 result=dict(status='ROM_BACKED_SENSOR_VERIFY_PIN_PROTOTYPE_PASS_NOT_PHYSICAL_RELEASE',cases=cases,
  gain_register=meta['gain_register'],inputs_sha256=sha(root/'inputs.json'),unmodified_official_models=True,hardware_access=False,board_release=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(result['status'],flush=True)

if __name__=='__main__':main()
