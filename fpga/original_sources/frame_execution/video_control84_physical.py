"""84 exact control factoring: original constraints and unchanged hard anchors."""
import json,pickle,shutil
from pathlib import Path
from run_units import ROOT,sha
from video_service_physical_profile import periphery_audit

BOARD=ROOT/'tmp/frame-execution/connected-board-video-control84-map1'
REFERENCE=ROOT/'tmp/frame-execution/video-service-place50-input-local'

def audit():
 p=BOARD/'control84-preplace-audit.json';d=json.loads(p.read_text())
 assert d['status']=='CONTROL84_SOURCE_CLOCK_TESTS_PREPLACE_PASS'
 for f,h in d['evidence'].items():assert sha(f)==h,f
 assert d['map_sha256']==sha(BOARD/'outflow/ti60f225_oob.map.v')
 assert d['vdb_sha256']==sha(BOARD/'outflow/ti60f225_oob.vdb')
 return d

def apply_profile(root,evidence):
 proof=audit();assert sha(root/'native_source.v')==sha(BOARD/'native_source.v')
 delta=periphery_audit(root);sdc=root/'ti60f225_oob.sdc';before=sha(sdc)
 assert before==sha(BOARD/'ti60f225_oob.sdc')
 canonical=REFERENCE/'ti60f225_oob.sdc'
 assert sha(canonical)=='464ef739d92e3fb1d2b15018863be99636efc1cae42ca581a0526f305c4be262'
 shutil.copy2(canonical,sdc)
 result=dict(profile='released_original_constraints_for_control84',before_sha256=before,after_sha256=sha(sdc),original_sdc=str(canonical),original_sdc_sha256=sha(canonical),
  unchanged_original_constraints=True,new_timing_exceptions=0,gray_bus_constraints=10,reset_periphery_audit=delta,
  current_nonempty_endpoints_pending=True,cdc_release=False,hardware_access=False,
  evidence={str(f):sha(f) for f in (Path(__file__),BOARD/'control84-preplace-audit.json',canonical)})
 (root/'service_clock_profile.json').write_text(json.dumps(result,indent=2));return result

def generate(root,reference):
 h,g,ps,drivers,loads=pickle.loads((BOARD/'integration-parse.pkl').read_bytes());assert h==sha(root/'outflow/ti60f225_oob.map.v')
 def loc(p):return {v[0]:tuple(map(int,v[1:])) for line in p.read_text().splitlines() if len(v:=line.split())==4 and not line.startswith('#')}
 target=root/'fixed_dsp.placeloc';current=loc(target);original=loc(REFERENCE/'fixed_dsp.placeloc')
 assert len(current)==288 and len(original)==404 and all(original[n]==xyz for n,xyz in current.items())
 assert all(g[n][0] in ('EFX_DSP24','EFX_DSP48') for n in current)
 added=set(original)-set(current)
 assert len(added)==116 and all(g[n][0] in ('EFX_RAM10','EFX_DPRAM10') for n in added)
 assert not any(n.startswith('bandwidth83_capture/') for n in original)
 shutil.copy2(REFERENCE/'fixed_dsp.placeloc',target)
 result=dict(status='RETAINED_RELEASED_HARD_ANCHORS_NO_ORDINARY_ANCHORS',candidate_map_sha256=h,assignments={n:list(x) for n,x in original.items()},
  total_hard_anchors=404,ordinary_anchors=0,new_capture_ram_automatic=7,assignments_sha256=sha(target),reference=str(REFERENCE),
  evidence={str(f):sha(f) for f in (Path(__file__),REFERENCE/'fixed_dsp.placeloc',BOARD/'control84-preplace-audit.json')},hardware_access=False)
 (root/'cnn_ram_placement.json').write_text(json.dumps(result,indent=2));return result

def verify_placed(root):
 proof=json.loads((root/'cnn_ram_placement.json').read_text())
 for f,h in proof['evidence'].items():assert sha(f)==h
 loc={v[0]:list(map(int,v[1:])) for line in (root/'outflow/ti60f225_oob.out.placeloc').read_text().splitlines() if len(v:=line.split())==4 and not line.startswith('#')}
 assert all(loc[n]==xyz for n,xyz in proof['assignments'].items())
 return dict(status='CONTROL84_404_HARD_ANCHORS_PLACED_PASS',count=404,ordinary_anchors=0,placement_sha256=sha(root/'outflow/ti60f225_oob.out.placeloc'))
