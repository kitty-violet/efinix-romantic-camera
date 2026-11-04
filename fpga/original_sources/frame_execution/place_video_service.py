"""Independent interface/pack/place, stopping for congestion review before route."""
import argparse,json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_command_repair import once

def audit_delta(board):
 meta=json.loads((board/'inputs.json').read_text());result=json.loads((board/'result.json').read_text())
 assert result['status']=='VIDEO_SERVICE_COMPLETE_NATIVE_SYNTHESIS_PROBE_PASS'
 assert sha(board/'inputs.json')==result['inputs_sha256']
 for r in meta['inputs']:assert sha(r['snapshot'])==r['sha256'],r['snapshot']
 graph=json.loads((board/'retained-map-audit.json').read_text())
 assert graph['status']=='VIDEO_SERVICE_RETAINED_MAPPED_GRAPH_PASS' and graph['retained']==58312
 assert graph['source_sha256']==sha(board/'native_source.v') and graph['map_sha256']==result['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v')
 assert result['vdb_sha256']==sha(board/'outflow/ti60f225_oob.vdb')
 from video_service_physical_profile import periphery_audit
 periphery_audit(board)
 return dict(status='SERVICE_MAP_BOUND_FOR_PLACEMENT_NOT_RELEASE',map_sha256=result['map_sha256'])

def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
 board=a.board.resolve();root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
 audit_delta(board)
 source=HERE/'probe_local_decode_place.py';code=source.read_text()
 code=once(code,'from local_control_decode import audit_delta','from place_video_service import audit_delta')
 code=once(code,"source=HERE/'probe_fourpll_reset_physical.py';code=source.read_text()", "source=HERE/'probe_fourpll_reset_physical.py';code=source.read_text().replace('from windows_process_tree import stop_launched_tree','from windows_process_tree_safe import stop_launched_tree')")
 code=once(code,"'from control_ddr_clock_profile import apply_profile'","'from video_service_physical_profile import apply_profile'")
 anchor="    code=once(code,'related_gray_layout=None\\n',"
 threads='''xml=root/'ti60f225_oob.xml';before_threads=xml.read_text()
needle='<efx:param name="max_threads" value="-1" value_type="e_integer" />'
assert before_threads.count(needle)==1
xml.write_text(before_threads.replace(needle,needle.replace('value="-1"','value="4"')),encoding='utf8')
for r in records:
 if Path(r['snapshot'])==xml:r['sha256']=sha(xml)
(root/'pnr_threads.json').write_text(json.dumps(dict(official_max_threads=4),indent=2))
optimization_change=None'''
 code=once(code,anchor,"    code=once(code,'optimization_change=None',"+repr(threads)+")\n"+anchor)
 code=once(code,'from audit_global_place import metrics','from camera_ratio_before_place import metrics')
 code=once(code,"ref=ROOT/'tmp/frame-execution/command-progress-seed3-1/pnr-seed1'","ref=ROOT/'tmp/frame-execution/continuous-srl-register-place1'")
 # Source-aware relative clock model is already applied by the profile.
 driver=root.parent/(root.name+'-wrapper.py');assert not driver.exists()
 compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
 args=[str(driver),'--root',str(root),'--base',str(board),'--cnn-ram-reference',str(ROOT/'tmp/frame-execution/native-local-driver-route1/canonical-seed3'),
  '--pnr-budget','1800','--logic-limit','60800','--fixed-dsp-layout','--clock-profile-proof',str(ROOT/'tmp/frame-execution/axi-valid-payload-physical1/clock_crossing_inventory.json'),'--optimization','CONGESTION_2','--seed','3']
 driver.with_suffix('.json').write_text(json.dumps(dict(argv=args,driver_sha256=sha(driver),builder_sha256=sha(__file__),source_sha256=sha(source),
  map_audit_sha256=sha(board/'retained-map-audit.json'),ordinary_anchors=0,hard_anchors=401,host_threads=4,route_started=False,hardware_access=False),indent=2))
 if a.prepare_only:print(json.dumps(dict(status='PLACEMENT_DRIVER_PREPARED',argv=args)));return
 sys.argv=args;runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':main()
