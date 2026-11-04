"""Run explicit repaired-candidate stages; immutable prior receipts retained."""
import json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from camera_checked_v2 import BOARD
import camera_checked_board as board_module
board_module.BOARD=BOARD

def main():
 action=sys.argv.pop(1)
 if action in ('place','shorts','prepare-route','route','audit','flow'):
  sys.argv=[str(HERE/'camera_checked_flow.py'),action,*sys.argv[1:]];runpy.run_path(str(HERE/'camera_checked_flow.py'),run_name='__main__');return
 source=HERE/('test_camera_checked_mapped.py' if action=='mapped' else 'test_camera_observer_host.py' if action=='observer' else 'review_camera_checked_place.py' if action=='review-place' else 'prepare_camera_route_review.py')
 assert action in ('mapped','observer','review-place','review-route');code=source.read_text()
 if action=='mapped':
  code=code.replace("('raw_stream_errors','inst_efx_csi2_rx", "('raw_stream_errors','raw_frontend/consumer/reservation_error','raw_frontend/consumer/fifo_error','inst_efx_csi2_rx")
 else:
  for a,b in (('camera-observer-host1','camera-observer-host2'),('camera-checked-place1','camera-checked-place2'),('camera-checked-evidence1','camera-checked-evidence2'),('camera-checked-route-review1','camera-checked-route-review2'),('camera-checked-bound1','camera-checked-bound2')):code=code.replace(a,b)
  code=code.replace('from finish_camera_checked_release import camera_evidence','from finish_camera_checked_release_v2 import camera_evidence')
  if action=='review-route':
   code=code.replace('56924/60800','CURRENT_REVIEW/60800').replace('adds292 XLR','adds the measured XLR delta')
   # Rationale supplied only after reading this new placement, not inherited.
   patch=ROOT/'tmp/frame-execution/camera-checked-route-rationale2.json';extra=json.loads(patch.read_text())
   code=code.replace('out.write_text(json.dumps(review,indent=2))',"review['rationale']="+repr(extra['rationale'])+"\n review['remaining_risks']="+repr(extra['remaining_risks'])+"\n review['reviewed_evidence']["+repr(str(patch))+"]="+repr(sha(patch))+"\n out.write_text(json.dumps(review,indent=2))")
 driver=ROOT/'tmp/frame-execution'/('camera-v2-'+action+'-driver.py');assert not driver.exists();compile(code,str(driver),'exec');driver.write_text(code)
 driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),driver_sha256=sha(driver),wrapper_sha256=sha(__file__),board=str(BOARD)),indent=2))
 sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')
if __name__=='__main__':main()
