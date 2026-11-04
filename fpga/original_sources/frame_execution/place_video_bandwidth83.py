"""One gated pack/place run; stop before routing for congestion review."""
import argparse,json,runpy,sys
from run_units import ROOT,sha
from video_bandwidth83_physical import audit,BOARD
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
from pathlib import Path
root=Path(a.root).resolve();assert root.is_relative_to(ROOT/'tmp/frame-execution') and not root.exists()
audit()
source=ROOT/'tmp/frame-execution/video-service-place50-input-local/physical_runner.py'
assert sha(source)=='b9b2d06c8b4ed11b551a2549fc136e75a35b68599371949e1f7e6acf69525932'
driver=root.with_name(root.name+'-driver.py');assert not driver.exists()
code=source.read_text()
for old,new in [('from video_service_physical_profile import apply_profile','from video_bandwidth83_physical import apply_profile'),('from input_isp_local_ram_locations import generate as retain_cnn_ram','from video_bandwidth83_physical import generate as retain_cnn_ram')]:
 assert code.count(old)==1;code=code.replace(old,new)
compile(code,str(driver),'exec');driver.write_text(code)
root.with_suffix('.launch.json').write_text(json.dumps(dict(status='BANDWIDTH83_GATED_PACK_PLACE_START_NO_ROUTE',seed=1,optimization='CONGESTION_2',logic_limit=60800,host_threads=4,hard_anchors=404,ordinary_anchors=0,hardware_access=False,
 evidence={str(f):sha(f) for f in (Path(__file__),source,driver,Path(__file__).with_name('video_bandwidth83_physical.py'),BOARD/'bandwidth83-preplace-audit.json')}),indent=2))
sys.argv=[str(driver),'--base',str(BOARD),'--root',str(root),'--pnr-budget','1800','--logic-limit','60800','--fixed-dsp-layout','--clock-profile-proof',str(ROOT/'tmp/frame-execution/axi-valid-payload-physical1/clock_crossing_inventory.json'),'--optimization','CONGESTION_2','--seed','1']
runpy.run_path(str(driver),run_name='__main__')
