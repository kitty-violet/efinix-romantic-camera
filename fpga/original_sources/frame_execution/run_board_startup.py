"""Execute a previously frozen/compiled full-board startup gate once."""
import argparse,json,os,re,subprocess,time
from pathlib import Path
from run_units import ROOT,sha
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--tag',default='run');p.add_argument('--timeout',type=int,default=900);a=p.parse_args();root=a.root.resolve()
 assert re.fullmatch(r'[a-z0-9_-]+',a.tag)
 result_name=a.tag+'_result.json';log_name=a.tag+'_simulation.log'
 assert root.is_relative_to(ROOT/'tmp') and not (root/result_name).exists()
 compiled=json.loads((root/'result.json').read_text());assert compiled['status']=='PREPARED_COMPILE_PASS'
 assert sha(root/'inputs.json')==compiled['inputs_sha256']
 meta=json.loads((root/'inputs.json').read_text())
 for row in meta['inputs']:assert sha(row['snapshot'])==row['sha256']
 assert sha(root/'tb_board_startup.sv')==meta['tb_sha256']
 assert sha(root/'csi_rx_controller_sim.sv')==meta['csi_simulation_sha256']
 assert sha(root/'ddr3_device_ID.vh')==meta['device_selection_header_sha256']
 env=os.environ.copy();env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 # Altera's official Known Differences permits continuing beyond recommended
 # capacity. Retain 14408 visibly as a warning; all functional errors stay fatal.
 cmd=['E:/quatra/questa_fse/win64/vsim.exe','-c','-warning','14408,16154','-voptargs=-warning 14408','work.tb_board_startup','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0']
 start=time.monotonic()
 try:
  with (root/log_name).open('wb') as log:
   proc=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT)
   try:proc.wait(timeout=a.timeout)
   except subprocess.TimeoutExpired:
    # Kill only this launched simulator and its descendants, not other tasks.
    subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True)
    proc.wait(timeout=30);raise
  body=(root/log_name).read_text(errors='replace')
  pass_marker='PASS DDR_UNIT_CALIBRATION' if meta.get('unit_ddr') else 'PASS BOARD_CALIBRATION'
  ok=proc.returncode==0 and pass_marker in body and not re.search(r'(?m)^#?\s*\*\* (Error|Fatal)|\bERROR:',body)
  result=dict(status='BOARD_CALIBRATION_PASS' if ok else 'BOARD_CALIBRATION_FAIL',returncode=proc.returncode,seconds=time.monotonic()-start,
   unit_ddr=meta.get('unit_ddr',False),markers=re.findall(r'^# ((?:PASS|PLL_SHIFT|BOARD_START|DDR_STATE|DDR_PIN|DDR_CALIBRATION_FLAG)[^\n]+)',body,re.M),errors=[x for x in body.splitlines() if '** Error' in x or '** Fatal' in x or 'ERROR:' in x],inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/log_name),full_board_frame_ms=None,board_release=False)
 except subprocess.TimeoutExpired:
  result=dict(status='BOARD_STARTUP_TIMEOUT',seconds=time.monotonic()-start,full_board_frame_ms=None,board_release=False)
 result['command']=cmd
 (root/result_name).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 if result['status']!='BOARD_CALIBRATION_PASS':raise SystemExit(1)
if __name__=='__main__':main()
