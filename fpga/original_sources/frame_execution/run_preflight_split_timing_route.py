"""Execute the exact preflighted driver with the existing project import path."""
import json,runpy,sys
from pathlib import Path
from run_units import ROOT,sha
p=ROOT/'tmp/board-test/split-timing-route-preflight1.json'
record=json.loads(p.read_text())
assert record['status']=='FULL_ROUTE_DRIVER_STATIC_GATES_PASS_NO_TOOL'
driver=Path(record['driver']);assert sha(driver)==record['driver_sha256']
assert not (ROOT/'tmp/frame-execution/split-outputs-timing-route1').exists()
sys.argv=[str(driver),*sys.argv[1:]]
runpy.run_path(str(driver),run_name='__main__')
