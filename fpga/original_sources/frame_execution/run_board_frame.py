"""Run prepared full-board frames; retain categorized DDR timing diagnostics."""
import argparse,collections,json,os,re,subprocess,time
from pathlib import Path
from run_units import ROOT,sha
from owned_simulator_process import spawn_owned,stop_owned_tree,close_owned
from host_runtime_budget import HostRuntimeBudget,MAX_HOST_SECONDS

def failure_kind(line):
 if 'Failed to open readmem' in line:return 'MISSING_INITIALIZATION_IMAGE'
 if ' ERROR: ' in line and 'io_model.memory' in line:return 'DDR_DEVICE_TIMING_ERROR'
 if '** Fatal:' in line or '** Error' in line:return 'SIMULATOR_ERROR'
 return None

def stop_spawned_tree(proc):
 """Stop only the still-running process created by this runner and its children."""
 return stop_owned_tree(proc)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--duration-us',type=int);p.add_argument('--optimize-level',type=int,choices=(4,5),default=4)
 p.add_argument('--host-timeout-seconds',type=int,default=MAX_HOST_SECONDS)
 a=p.parse_args();root=a.root.resolve()
 if not 0<a.host_timeout_seconds<=MAX_HOST_SECONDS:p.error('Host timeout must be between 1 and 14400 seconds')
 meta=json.loads((root/'inputs.json').read_text());native=Path(meta['native']) if meta.get('native') else None
 prepared=json.loads((root/'prepared.json').read_text())
 assert prepared['status']=='BOARD_FRAME_COMPILE_PASS' and prepared['inputs_sha256']==sha(root/'inputs.json')
 assert sha(root/'tb_board_startup.sv')==meta['tb_sha256']
 for row in meta.get('inputs',[]):assert sha(row['snapshot'])==row['sha256'],row['snapshot']
 env=os.environ.copy();env.update(SALT_LICENSE_SERVER='E:/quatra/LR-195771_License.dat',PATH='E:/fpga/tools/verilator/ucrt64/bin;'+env['PATH'])
 # Official Starter Known Differences permits continuing above recommended
 # capacity. Keep those diagnostics visible; functional errors remain fatal.
 cmd=['E:/quatra/questa_fse/win64/vsim.exe','-c','-warning','14408,16154',f'-voptargs=-warning 14408 -O{a.optimize_level}']
 if native:
  if meta.get('native_kind')=='production_cnn':
   from production_board_binding import validate
   validate(root,meta)
   assert Path(meta['native_library'])==native/'production'
   cmd+=['-sv_lib',(native/'production').as_posix()]
  else:cmd+=['-sv_lib',(native/'obj/libboardcnn').as_posix()]
 # onerror/onbreak operate inside a DO macro, not a string of interactive
 # commands. Keep the textual failure audit as a second independent check.
 (root/'run_board.do').write_text('onerror {quit -code 1}\nonbreak {quit -code 1}\n'+
     (f'run {a.duration_us} us' if a.duration_us else 'run -all')+'\nquit -code 0\n')
 cmd+=['work.tb_board_startup','-do','run_board.do']
 start=time.monotonic();proc=spawn_owned(cmd,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace',bufsize=1)
 (root/'running.json').write_text(json.dumps(dict(pid=proc.pid,command=cmd,inputs_sha256=sha(root/'inputs.json')),indent=2))
 budget=HostRuntimeBudget(proc,a.host_timeout_seconds,root,start).start()
 errors=collections.Counter();events=[];passed=False;fail=[];progress=[];last_console_progress=-10.0;first_abort=None
 suffix=('_'+str(a.duration_us)+'us')if a.duration_us else ''
 with (root/('simulation'+suffix+'.log')).open('w',encoding='utf8')as log:
  for line in proc.stdout:
   if ' ERROR: ' in line and 'io_model.memory' in line:
    key=line.split('ERROR:',1)[1].strip();errors[key]+=1
    if errors[key]>12:continue
   log.write(line);log.flush()
   if re.search(r'BOARD_PROGRESS|BOARD_BOOT_PHASE|CAMERA_(FRAME|PORT)_BEGIN|RAW_FRAME_(BEGIN|STORED|READ_DONE)|RGB_FRAME_(BEGIN|COMPLETE)|CNN_BEGIN|CNN_COMMIT|SCREEN_BEGIN|SCREEN_COMPLETE|BOARD_CPU_RUN|BOARD_CAL_DONE|PAYLOAD_PRELOADED',line):
    elapsed=time.monotonic()-start
    events.append(line.strip())
    # Keep every event in the evidence log/JSON. Throttle only repetitive
    # console progress so an unattended terminal cannot stall the simulator.
    if 'BOARD_PROGRESS' not in line or elapsed-last_console_progress>=10:
     print(line.strip(),flush=True)
     last_console_progress=elapsed
    point=dict(wall_seconds=round(elapsed,3),event=line.strip())
    progress.append(point)
    (root/'progress.json').write_text(json.dumps(point,indent=2))
   if 'PASS BOARD_THREE_FRAMES_TMDS' in line or (meta.get('raw_diagnostic') and 'PASS RAW_DIAGNOSTIC_FIRST_LINES' in line):passed=True
   if '** Fatal:' in line or '** Error' in line:fail.append(line.strip())
   kind=failure_kind(line)
   if kind and first_abort is None and budget.result is None:
    first_abort=dict(kind=kind,line=line.strip(),wall_seconds=round(time.monotonic()-start,3),
                     reason='First confirmed simulation error; stop invalid long run immediately.')
    # Save the reason before terminating the owned simulator tree. Preserve
    # subsequent shutdown text, but never mistake it for the original cause.
    (root/'failfast.json').write_text(json.dumps(first_abort,indent=2))
    print('BOARD_FAIL_FAST '+json.dumps(first_abort),flush=True)
    first_abort['termination']=stop_spawned_tree(proc)
    (root/'failfast.json').write_text(json.dumps(first_abort,indent=2))
 ret=proc.wait()
 host_abort=budget.finish()
 close_owned(proc)
 for row in meta.get('inputs',[]):assert sha(row['snapshot'])==row['sha256'],row['snapshot']
 complete=passed and not fail and first_abort is None and host_abort is None and ret==0
 result=dict(status=('BOARD_DIGITAL_MEASUREMENT_COMPLETE' if not errors else 'BOARD_FUNCTIONAL_MEASUREMENT_COMPLETE_PHY_FAIL') if complete else 'BOARD_FRAME_RUN_FAILED',
    returncode=ret,seconds=time.monotonic()-start,events=events,progress=progress,failures=fail,ddr_model_error_counts=dict(errors),
    ddr_model_error_total=sum(errors.values()),first_error_abort=first_abort,timing_gate_pass=False,physical_pass=False,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/('simulation'+suffix+'.log')))
 result['host_time_budget_seconds']=a.host_timeout_seconds
 result['host_budget_abort']=host_abort
 if a.duration_us:result.update(status='BOUNDED_SPEED_PROBE_NOT_FULL_FRAME' if ret==0 and not fail else 'SPEED_PROBE_FAILED',simulated_us=a.duration_us)
 if meta.get('raw_diagnostic'):result.update(status='RAW_DIAGNOSTIC_PASS' if complete and not errors else 'RAW_DIAGNOSTIC_FAIL',full_board_frame=False)
 if host_abort is not None:result.update(status='BOARD_FRAME_HOST_BUDGET_EXCEEDED',complete_frame=False,full_board_frame=False)
 (root/('run_result'+suffix+'.json')).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('events','progress')}),flush=True)
 if ret!=0 or fail or errors or first_abort or host_abort or (not passed and not a.duration_us):raise SystemExit(1)
if __name__=='__main__':main()
