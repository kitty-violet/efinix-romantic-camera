"""Run a frozen native comparison with the installed Questa, bounded and offline."""
import argparse,json,os,re,shutil,subprocess,time
from pathlib import Path
from run_units import ROOT,sha
from windows_process_tree_safe import stop_launched_tree

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 src=a.source.resolve();root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
 meta=json.loads((src/'inputs.json').read_text())
 for f,h in meta['files'].items():assert sha(f)==h,f
 root.mkdir();sim=Path('E:/quatra/questa_fse/win64')
 for f in src.iterdir():
  if f.suffix in ('.v','.sv'):shutil.copy2(f,root/f.name)
 files={str(f):sha(f) for f in root.iterdir()}
 manifest=dict(files=files,parent_inputs=str(src/'inputs.json'),parent_inputs_sha256=sha(src/'inputs.json'),runner_sha256=sha(__file__),simulator=str(sim),hardware_access=False)
 (root/'inputs.json').write_text(json.dumps(manifest,indent=2))
 env=os.environ.copy();env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 start=time.monotonic();result=dict(status='RUNNING',hardware_access=False,board_release=False)
 def invoke(args,name,budget):
  with (root/name).open('x') as f:
   child=subprocess.Popen(list(map(str,args)),cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
   try:rc=child.wait(timeout=budget)
   except subprocess.TimeoutExpired:
    (root/(name+'.stop.json')).write_text(json.dumps(stop_launched_tree(child),indent=2));raise
  body=(root/name).read_text();assert rc==0 and not re.search(r'^#?\s*\*\* (Error|Fatal)',body,re.M),body[-2500:]
  return body
 try:
  invoke([sim/'vlib.exe','work'],'vlib.log',20)
  invoke([sim/'vlog.exe','-sv','-timescale','1ns/1ps',*root.glob('*.v'),*root.glob('*.sv')],'compile.log',60)
  body=invoke([sim/'vsim.exe','-c','work.tb_video_snapshot_serializer','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],'simulate.log',240)
  assert 'PASS VIDEO_SERVICE_INDEXED_SERIAL_NATIVE' in body
  for f,h in files.items():assert sha(f)==h
  result.update(status='VIDEO_SERVICE_INDEXED_SERIAL_NATIVE_PASS',markers=re.findall(r'^# PASS.*$',body,re.M))
 except Exception as e:result.update(status='FAIL',error=repr(e));raise
 finally:
  result.update(seconds=time.monotonic()-start,inputs_sha256=sha(root/'inputs.json'))
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
