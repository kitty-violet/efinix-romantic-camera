"""Reuse existing bounded implementation gates for the read-only OFP4 delta."""
import json,sys,subprocess,re
from pathlib import Path
from run_units import ROOT,HERE,sha
from fault_probe_board import audit_delta,BASE
FAST_TCK=False

def proof(board,folder):
 audit_delta(board);r=json.loads((folder/'result.json').read_text());i=json.loads((folder/'inputs.json').read_text())
 assert r['status']=='FAULT_PROBE_CLOCK_RESET_SERIAL_AND_REAL_ADAPTER_PASS'
 assert r['inputs_sha256']==sha(folder/'inputs.json') and i['board_inputs_sha256']==sha(board/'inputs.json')
 for n,h in i['files'].items():assert sha(folder/n)==h
 for x in r['results']:assert sha(folder/x['log'])==x['sha256'] and x['passes']
 return dict(path=str(folder),result_sha256=sha(folder/'result.json'))

def adjust(code):
 # Operates after the established OFP3 tests have been generated. Their
 # command, APB, idle, reset and error checks remain, with a longer scan only.
 code=code.replace('from read_control_ddr_progress import decode','from read_fault_probe import decode')
 code=code.replace('[449:0]','[769:0]').replace('[449:66]','[769:66]')
 code=re.sub(r'(?<!\d)450(?!\d)','770',code).replace("32'h4f465033","32'h4f465034").replace('raw[49:42]!==3','raw[49:42]!==4')
 code=code.replace('wire [383:0] payload={ddr_lock_drops','wire [703:0] payload={320\'h123456789abcdef,ddr_lock_drops')
 code=code.replace('len(selected)<6000','len(selected)<7500')
 if "source=HERE/'validate_global_mapped_startup.py'" in code:
  # Keep the adapter's complete diagnostic bus as an external state boundary
  # in the CPU-only cone. Otherwise the old dependency-name heuristic keeps
  # just some bits while cutting others, producing duplicate bus drivers.
  owned="def owned(n):\n    if n.startswith('\\\\adapter_diagnostic['):return False"
  inject="code=code.replace('def owned(n):',"+repr(owned)+")\n    "
  anchor=" code=code.replace(r'\\cpu/sequencer/state ',r'\\cpu/boot_observer_state ')"
  assert code.count(anchor)==1
  code=code.replace(anchor," code=code.replace(marker,"+repr(inject)+"+marker)\n"+anchor)
  code=code.replace(anchor," code=code.replace(\"'simulation.log',90)\",\"'simulation.log',180)\")\n"+anchor)
 code=code.replace('#3000000;$fatal(1,"timeout")','#5000000;$fatal(1,"timeout")')
 code=code.replace('#300000;$fatal(1,"DDR observation simulation limit")','#500000;$fatal(1,"DDR observation simulation limit")')
 if 'simulation_only_inline' in code:
  code=code.replace('from inline_official_state_luts import inline','from inline_fault_probe_luts import inline')
  code=code.replace("H/'inline_official_state_luts.py'","H/'inline_fault_probe_luts.py'")
  # Three snapshots each use three complete scans: 9*770us > old5ms.
  code=code.replace('#5000000;$fatal(1,"simulation deadline")','#10000000;$fatal(1,"simulation deadline")')
  if FAST_TCK:
   before='task tick;begin #500;TCK=1;#500;TCK=0;end endtask'
   assert code.count(before)==1 and code.count("'tck_hz':1000000")==1
   code=code.replace(before,'task tick;begin #50;TCK=1;#50;TCK=0;end endtask')
   code=code.replace("'tck_hz':1000000","'tck_hz':10000000,'stimulus_change':'10MHz TCK at original constrained maximum; same commands, scan bits, quiet cycles and reset assertions'")
 code=code.replace("['vsim.exe','-c','work.tb'","['vsim.exe','-c','-warning','14408,16154','-voptargs=-warning 14408','work.tb'")
 return code

def main():
 global FAST_TCK
 FAST_TCK='--tck-fast' in sys.argv
 if FAST_TCK:sys.argv.remove('--tck-fast')
 import cached_head_flow as flow
 import local_control_decode
 import control_ddr_combined,validate_control_ddr
 mode=sys.argv[1]
 if mode in ('rtl','startup','observe','serial'):
  Path(sys.argv[sys.argv.index('--root')+1]).resolve().parent.mkdir(parents=True,exist_ok=True)
 if mode=='serial' and '--modelsim' in sys.argv:
  sys.argv.remove('--modelsim')
  # All mapped cone cells here are public official primitives. The installed
  # ModelSim handles those without Questa Starter's capacity slowdown; refuse
  # any encrypted source rather than substituting an implementation model.
  from run_units import MS
  popen=subprocess.Popen
  def public_launch(cmd,*pos,**kw):
   name=Path(str(cmd[0])).name.lower()
   if name in ('vlib.exe','vlog.exe','vsim.exe'):
    cwd=Path(kw['cwd']);files=list(cwd.glob('*.v'))+list(cwd.glob('*.sv'))
    assert files and all('pragma protect' not in f.read_text(errors='replace') for f in files)
    args=[];skip=False
    for x in cmd[1:]:
     if skip:skip=False;continue
     if x=='-warning':skip=True;continue
     if str(x).startswith('-voptargs='):continue
     args.append(x)
    cmd=[str(MS/name),*args]
    record=cwd/'simulator_override.json'
    data=dict(simulator=str(MS),public_official_models_only=True,files={f.name:sha(f) for f in files},adapter_sha256=sha(__file__),hardware_access=False)
    if record.exists():assert json.loads(record.read_text())==data
    else:record.write_text(json.dumps(data,indent=2))
   return popen(cmd,*pos,**kw)
  subprocess.Popen=public_launch
 flow.audit_delta=audit_delta;flow.BASE=BASE;flow.proofs=proof
 local_control_decode.audit_delta=audit_delta;control_ddr_combined.audit_delta=audit_delta
 validate_control_ddr.audit_delta=audit_delta
 if mode in ('rtl','startup','serial'):
  original=validate_control_ddr.driver_run
  validate_control_ddr.driver_run=lambda source,code,root,args:original(source,adjust(code),root,args)
 if mode=='observe':
  # This gate is standalone rather than nested via driver_run.
  import runpy
  source=HERE/'validate_control_ddr_observation.py'
  code=adjust(source.read_text())
  root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
  driver=root.parent/(root.name+'-ofp4-driver.py');assert not driver.exists()
  driver.write_text(code);sys.argv=[str(driver),*sys.argv[2:]]
  runpy.run_path(str(driver),run_name='__main__');return
 if mode=='route':
  import runpy
  source=HERE/'route_local_decode_board.py';code=source.read_text()
  assert code.count("HERE/'audit_command_progress_gray.py'")==1
  code=code.replace("HERE/'audit_command_progress_gray.py'","HERE/'audit_fault_probe_gray.py'")
  code=code.replace('from inline_official_state_luts import inline','from inline_fault_probe_luts import inline')
  code=code.replace("HERE/'inline_official_state_luts.py'","HERE/'inline_fault_probe_luts.py'")
  before="            except subprocess.TimeoutExpired:save(root/(name+'-timeout.json'),stop_launched_tree(proc));raise"
  after="            except subprocess.TimeoutExpired:\n                save(root/(name+'-timeout.json'),stop_launched_tree(proc))\n                row.update(status='timed_out_process_tree_stopped',seconds=time.monotonic()-start)\n                save(root/'stages.json',stages);raise"
  assert code.count(before)==1;code=code.replace(before,after)
  root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
  driver=root.parent/(root.name+'-driver.py');assert not driver.exists()
  driver.write_text(code,encoding='utf8')
  driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),driver_sha256=sha(driver),wrapper_sha256=sha(__file__)),indent=2))
  sys.argv=[str(driver),*sys.argv[2:]];runpy.run_path(str(driver),run_name='__main__');return
 flow.main()
if __name__=='__main__':main()
