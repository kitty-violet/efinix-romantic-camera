"""Bounded current-map checks for the selected balanced-ready candidate.

Runs after boot proof, reuses its parsed map, and stops on the first error.
Never performs synthesis, routing or hardware access.
"""
import argparse,json,sys,time
from pathlib import Path
from run_units import HERE,ROOT,sha
from map_video_service_safe import run

def main():
 p=argparse.ArgumentParser()
 for n in ('board','native','boot','out','fixture'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();board=a.board.resolve();native=a.native.resolve();boot=a.boot.resolve();out=a.out.resolve()
 assert all(x.is_relative_to(ROOT/'tmp') for x in (board,native,boot,out,a.fixture.resolve()))
 assert not out.exists()
 assert json.loads((boot/'result.json').read_text())['status']=='VIDEO_SERVICE_ACTUAL_MAP_BOOT_PASS'
 out.mkdir();stages=[];start=time.monotonic()
 def check(name,args,budget=240):
  _,seconds=run([sys.executable,HERE/name,*args],out,name+'.log',budget)
  stages.append(dict(script=name,seconds=seconds,log_sha256=sha(out/(name+'.log'))))
  (out/'stages.json').write_text(json.dumps(stages,indent=2))
 try:
  warning=boot/'warning-port-audit.json'
  if warning.exists():
   record=json.loads(warning.read_text())
   assert record['status']=='EVERY_MISSING_PORT_UNUSED_OUTPUT_OR_BOOLEAN_INDEPENDENT_INPUT_PASS'
   for f,h in record['evidence'].items():assert sha(f)==h,f
  else:check('audit_video_service_mapped_boot_warnings.py',['--root',boot])
  check('audit_video_service_protected_warnings.py',['--board',board,'--boot',boot])
  for script,filename in [('validate_video_service_narrow_mapped_pages.py','pages-proof.json'),
                          ('validate_video_service_narrow_mapped_epoch.py','nested-epoch-proof.json'),
                          ('audit_video_service_narrow_cdc.py','service-cdc-inventory.json')]:
   check(script,['--board',board,'--native',native,'--out',board/filename])
  check('validate_video_service_narrow_status_selection.py',['--board',board,'--old',ROOT/'tmp/frame-execution/connected-board-video-service-probe18','--output',board/'status-selection-proof.json'])
  check('validate_video_service_narrow_counter.py',['--board',board,'--output',board/'sweep-counter-proof.json'])
  check('prepare_video_service_narrow_mapped_serial.py',['--board',board,'--reference',ROOT/'tmp/frame-execution/video-service-narrow-fixture1','--root',a.fixture.resolve()])
  result=dict(status='CURRENT_BALANCED_MAP_SHORT_CHECKS_PASS',stages=stages,hardware_access=False,board_release=False)
 except Exception as e:
  result=dict(status='FAIL',error=repr(e),stages=stages,hardware_access=False,board_release=False);raise
 finally:
  result.update(seconds=time.monotonic()-start,runner_sha256=sha(__file__))
  (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
