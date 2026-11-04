"""Offline observer RTL, mapped one-shot gates, and all packed host bits."""
import json,os,shutil,subprocess
from run_units import ROOT,HERE,sha
from camera_checked_board import BOARD,audit_delta
from camera_checked_cdc import map_structure
from read_camera_checked import decode

def main():
 out=ROOT/'tmp/board-test/camera-observer-host1';assert not out.exists();out.mkdir();audit_delta(BOARD)
 slots=json.loads((BOARD/'camera_graft.json').read_text())['diagnostic_slots']
 def raw(v):
  wire=sum(((v>>i)&1)<<(896+s) for i,s in enumerate(slots))
  return (wire|0x4f465035|(5<<40)|(1<<32))<<2
 cfg=0xf00a3e092a2acd6b
 sample=cfg|(480<<64)|(1080<<80)|(7<<96)|(0x5a5<<99)|(1<<110)
 d=decode(raw(sample),slots)['camera']
 assert d['config_verified'] and d['chip_id']=='0xcd6b' and d['register']=='0x3e09'
 assert d['first_complete_line_words']==480 and d['first_complete_frame_lines']==1080
 assert d['raw_error_bits']==0x5a5 and d['boundary_error'] and d['seen_pixel_valid'] and d['seen_hsync'] and d['seen_vsync_transition']
 checks=0
 for i in range(111):
  value=raw(1<<i);wire=(value>>2)>>896
  assert sum(((wire>>s)&1)<<j for j,s in enumerate(slots))==1<<i
  decode(value,slots);checks+=1
 for v in (0x1180310700cd0000,0x450432098180cd6b,0x1200310700cd0000):
  r=decode(raw(v),slots)['camera'];assert not r['config_verified'] and r['error_code']>0;checks+=1
 for bad in (slots[:-1],[0]*111,slots[:-1]+[128]):
  try:decode(raw(sample),bad)
  except ValueError:checks+=1
  else:raise AssertionError('bad slot map accepted')
 structure=map_structure(BOARD);(out/'mapped_structure.json').write_text(json.dumps(structure,indent=2))
 for n in ('camera_input_observer.sv','tb_camera_input_observer.sv'):shutil.copy2(HERE/n,out/n)
 tool=ROOT/'tools/iverilog/ucrt64/bin';env=os.environ.copy();env['PATH']=str(tool)+os.pathsep+env.get('PATH','')
 cmd=[str(tool/'iverilog.exe'),'-g2012','-s','tb_camera_input_observer','-o','sim.vvp','camera_input_observer.sv','tb_camera_input_observer.sv']
 r=subprocess.run(cmd,cwd=out,env=env,capture_output=True,text=True,timeout=30);(out/'compile.log').write_text(r.stdout+r.stderr);assert r.returncode==0
 r=subprocess.run([str(tool/'vvp.exe'),'sim.vvp'],cwd=out,env=env,capture_output=True,text=True,timeout=30);(out/'simulation.log').write_text(r.stdout+r.stderr)
 assert r.returncode==0 and 'PASS CAMERA_OBSERVER' in r.stdout
 inputs={str(f):sha(f) for f in [HERE/'read_camera_checked.py',HERE/'camera_checked_cdc.py',HERE/'test_camera_observer_host.py',BOARD/'camera_graft.json',BOARD/'outflow/ti60f225_oob.map.v',out/'camera_input_observer.sv',out/'tb_camera_input_observer.sv']}
 result=dict(status='CAMERA_OBSERVER_RTL_MAPPED_GATES_AND_HOST_PASS',map_sha256=sha(BOARD/'outflow/ti60f225_oob.map.v'),host_checks=checks,enable_truth_checks=structure['capture_enable_truth_checks'],held_bits=95,
  example=d,evidence={**inputs,str(out/'mapped_structure.json'):sha(out/'mapped_structure.json'),str(out/'simulation.log'):sha(out/'simulation.log')},hardware_access=False)
 (out/'result.json').write_text(json.dumps(result,indent=2));print(result['status'],checks)
if __name__=='__main__':main()
