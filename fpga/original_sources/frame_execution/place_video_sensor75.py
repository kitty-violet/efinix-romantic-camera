"""Use the proven input-ISP placement recipe after sensor-specific gates."""
import argparse,json,runpy,sys
from pathlib import Path
from run_units import ROOT,sha

def audit_delta(board):
 proof=json.loads((board/'sensor75-map-audit.json').read_text());assert proof['status']=='SENSOR75_EXACT_SOURCE_SCOPE_FRESH_MAP_PASS'
 for f,h in proof['evidence'].items():assert sha(f)==h
 assert proof['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v') and proof['vdb_sha256']==sha(board/'outflow/ti60f225_oob.vdb')
 from video_service_physical_profile import periphery_audit
 periphery_audit(board)
 cdc=json.loads((board/'sensor75-cdc-delta.json').read_text());assert cdc['status']=='SENSOR75_CDC_DELTA_CLASSIFIED_NOT_ROUTED_PASS' and cdc['map_sha256']==proof['map_sha256']
 for f,h in cdc['evidence'].items():assert sha(f)==h
 warnings=json.loads((board/'sensor75-warning-review.json').read_text());assert warnings['status']=='SENSOR75_WARNINGS_REVIEWED_NO_NEW_UNCLASSIFIED' and warnings['map_sha256']==proof['map_sha256']
 for f,h in warnings['evidence'].items():assert sha(f)==h
 return proof

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--pin-test',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 board=a.board.resolve();pin=a.pin_test.resolve();root=a.root.resolve()
 assert root.is_relative_to(ROOT/'tmp/frame-execution') and not root.exists()
 proof=audit_delta(board)
 tested=json.loads((pin/'result.json').read_text());assert tested['status'] in ('SENSOR75_MAPPED_I2C_PINS_PASS','SENSOR75_EQUIVALENT_MAPPED_I2C_PINS_EVIDENCE_PASS')
 inputs=json.loads((pin/'inputs.json').read_text());assert tested['inputs_sha256']==sha(pin/'inputs.json')
 prepared=Path(inputs['prepared']);assert inputs['prepared_sha256']==sha(prepared/'inputs.json')
 assert json.loads((prepared/'inputs.json').read_text())['map_sha256']==proof['map_sha256']
 for f,h in inputs['files'].items():assert sha(f)==h
 source=ROOT/'tmp/frame-execution/video-service-place50-input-local/physical_runner.py'
 assert sha(source)=='b9b2d06c8b4ed11b551a2549fc136e75a35b68599371949e1f7e6acf69525932'
 driver=root.with_name(root.name+'-driver.py');assert not driver.exists()
 code=source.read_text()
 assert code.count('from video_service_physical_profile import apply_profile')==1
 code=code.replace('from video_service_physical_profile import apply_profile','from video_sensor75_physical import apply_profile')
 assert code.count('from input_isp_local_ram_locations import generate as retain_cnn_ram')==1
 code=code.replace('from input_isp_local_ram_locations import generate as retain_cnn_ram','from video_sensor75_physical import generate as retain_cnn_ram')
 compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
 record=root.with_suffix('.launch.json')
 record.write_text(json.dumps(dict(status='SENSOR75_GATED_PACK_PLACE_START_NO_ROUTE',
  evidence={str(f):sha(f) for f in (Path(__file__),source,driver,Path(__file__).with_name('video_sensor75_physical.py'),board/'sensor75-map-audit.json',board/'sensor75-cdc-delta.json',board/'sensor75-warning-review.json',pin/'inputs.json',pin/'result.json')},
  seed=3,optimization='CONGESTION_2',logic_limit=60800,host_threads=4,
  ordinary_anchors=0,hard_anchors=404,hardware_access=False),indent=2))
 sys.argv=[str(driver),'--base',str(board),'--root',str(root),'--pnr-budget','1800','--logic-limit','60800','--fixed-dsp-layout',
  '--clock-profile-proof',str(ROOT/'tmp/frame-execution/axi-valid-payload-physical1/clock_crossing_inventory.json'),'--optimization','CONGESTION_2','--seed','3']
 runpy.run_path(str(driver),run_name='__main__')
