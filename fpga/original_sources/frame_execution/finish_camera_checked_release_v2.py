"""Repaired camera wiring: same complete release gates, fresh receipts."""
import inspect
from finish_camera_checked_release import *
from camera_checked_v2 import BOARD
SHORTS=ROOT/'tmp/frame-execution/camera-checked-evidence4'
RTL=SHORTS/'rtl';SERIAL=SHORTS.parent/(SHORTS.name+'-serial/new')
MAP_AUDIT=SHORTS/'map-audit';ADAPTER=SHORTS.parent/(SHORTS.name+'-adapter')
import finish_camera_checked_release as original
for name in ('parent_evidence','warning_audit','camera_evidence','extra_evidence','package_code','main'):
 code=inspect.getsource(getattr(original,name))
 if name=='camera_evidence':
  code=code.replace("(('camera-checked-mapped-questa2',[4]),('camera-checked-mapped-questa3',[0,6,8]))","(('camera-checked-mapped-questa4',[0,4,6,8]),)")
  code=code.replace('camera-observer-host1','camera-observer-host2')
  code=code.replace(' return dict(rtl_cases='," host=bind(ROOT/'tmp/board-test/camera-reader-offline1/result.json');assert host['status']=='CAMERA_HOST_FRESHNESS_TIMEOUT_FORMAT_PASS';verify(host['evidence'])\n return dict(rtl_cases=")
 if name=='extra_evidence':
  code=code.replace(' camera_evidence(bind,verify)'," camera_evidence(bind,verify)\n wiring=bind(board/'camera_wiring_inputs.json');verify(wiring['evidence'])\n external=bind(board/'camera_external_input_audit.json');assert external['status']=='ALL_CAMERA_MAPPED_INPUTS_DRIVEN_RAW6_OR_RAW5_ZERO_PASS';verify(external['evidence'])\n verify({str(HERE/'camera_checked_v2.py'):bind(board/'inputs.json')['camera_wiring_repair_sha256']})")
 if name=='package_code':
  code=code.replace("return source,code","return source,code.replace('finish_camera_checked_release','finish_camera_checked_release_v2')")
 if name=='main':
  code=code.replace("'finish_camera_checked_release.py'","'finish_camera_checked_release_v2.py'")
 exec(compile(code,__file__+':'+name,'exec'),globals())
if __name__=='__main__':main()
