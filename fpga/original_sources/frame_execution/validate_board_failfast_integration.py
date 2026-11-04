"""Exercise the real board runner's report path with a Python-only simulator fake."""
import argparse,json,subprocess,sys,time
from pathlib import Path
import run_board_frame as runner
from owned_simulator_process import spawn_owned
from run_units import ROOT,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
 cases={
  'normal':"print('# ** Warning: optional port',flush=True); print('# PASS BOARD_THREE_FRAMES_TMDS',flush=True)",
  'ddr_error':"print('# tb_board_startup.io_model.memory.chk_err: at time 99 ps ERROR: tWTR violation',flush=True); time.sleep(60); print('SHOULD_NEVER_REACH',flush=True)",
  'fatal':"print('# ** Fatal: test-only injected failure',flush=True); time.sleep(60); print('SHOULD_NEVER_REACH',flush=True)"}
 rows=[];original_argv=sys.argv;original_spawn=runner.spawn_owned
 try:
  for name,code in cases.items():
   case=root/name;case.mkdir();(case/'tb_board_startup.sv').write_text('// TEST ONLY: no FPGA design\n')
   (case/'inputs.json').write_text(json.dumps(dict(native=None,tb_sha256=sha(case/'tb_board_startup.sv'),test_only=True)))
   (case/'prepared.json').write_text(json.dumps(dict(status='BOARD_FRAME_COMPILE_PASS',inputs_sha256=sha(case/'inputs.json'))))
   def fake_spawn(command,**kwargs):
    return spawn_owned([sys.executable,'-u','-c','import time; '+code],**kwargs)
   runner.spawn_owned=fake_spawn;sys.argv=['run_board_frame.py','--root',str(case)];start=time.monotonic();exit_code=0
   try:runner.main()
   except SystemExit as exc:exit_code=exc.code
   elapsed=time.monotonic()-start;result=json.loads((case/'run_result.json').read_text())
   assert sha(case/'simulation.log')==result['log_sha256'] and elapsed<10
   if name=='normal':
    assert exit_code==0 and result['status']=='BOARD_DIGITAL_MEASUREMENT_COMPLETE' and result['first_error_abort'] is None
   else:
    assert exit_code==1 and result['status']=='BOARD_FRAME_RUN_FAILED'
    assert result['first_error_abort']['kind']==('DDR_DEVICE_TIMING_ERROR' if name=='ddr_error' else 'SIMULATOR_ERROR')
    assert 'SHOULD_NEVER_REACH' not in (case/'simulation.log').read_text()
    assert result['ddr_model_error_total']==(1 if name=='ddr_error' else 0)
   rows.append(dict(case=name,seconds=elapsed,expected_status=result['status'],passed=True))
 finally:runner.spawn_owned=original_spawn;sys.argv=original_argv
 report=dict(status='BOARD_RUNNER_FAILFAST_REPORT_INTEGRATION_PASS',test_only=True,actual_eda_invoked=False,cases=rows,
  runner_sha256=sha(Path(runner.__file__)))
 (root/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
