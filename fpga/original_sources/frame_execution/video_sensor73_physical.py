"""Frozen released clock constraints and hard anchors for the scoped sensor edit."""
import json,pickle,shutil
from pathlib import Path
from run_units import ROOT,sha
from complete_native_logic_split import protected_blocks
from video_service_physical_profile import periphery_audit

REFERENCE=ROOT/'tmp/frame-execution/video-service-place50-input-local'
BOARD=ROOT/'tmp/frame-execution/connected-board-video-sensor73-map2'
PARENT=ROOT/'tmp/frame-execution/connected-board-video-service-probe60-input-isp-explicit'

def apply_profile(root,evidence):
 assert sha(root/'native_source.v')==sha(BOARD/'native_source.v')
 proof=json.loads((BOARD/'sensor73-map-audit.json').read_text());assert proof['status']=='SENSOR73_EXACT_SOURCE_SCOPE_FRESH_MAP_PASS'
 parentmap=PARENT/'outflow/ti60f225_oob.map.v'
 assert sha(parentmap)=='21bfdc8a33626feed493cddc74fd73f7c191c7ad33346c2c520f6a6582407c26'
 assert protected_blocks((root/'native_source.v').read_text())==protected_blocks((PARENT/'native_source.v').read_text())
 delta=periphery_audit(root);before=root/'ti60f225_oob.sdc';oldsha=sha(before)
 assert oldsha==sha(PARENT/'ti60f225_oob.sdc')
 canonical=REFERENCE/'ti60f225_oob.sdc'
 assert sha(canonical)=='464ef739d92e3fb1d2b15018863be99636efc1cae42ca581a0526f305c4be262'
 pmeta=json.loads((REFERENCE/'physical_inputs.json').read_text())
 assert sha(canonical)==next(r['sha256'] for r in pmeta['inputs'] if Path(r['snapshot'])==canonical)
 shutil.copy2(canonical,before)
 result=dict(profile='released69_exact_original_SDC_for_full_native_sensor73',before_sha256=oldsha,after_sha256=sha(before),
  original_sdc=str(canonical),original_sdc_sha256=sha(canonical),original_physical_inputs_sha256=sha(REFERENCE/'physical_inputs.json'),
  unchanged_original_constraints=True,new_timing_exceptions=0,gray_bus_constraints=10,reset_periphery_audit=delta,
  protected_input_byte_identical_to_released_original_source=True,protected_source=str(PARENT/'native_source.v'),protected_source_sha256=sha(PARENT/'native_source.v'),
  current_nonempty_endpoints_pending=True,cdc_release=False,hardware_access=False,
  evidence={str(f):sha(f) for f in (Path(__file__),BOARD/'sensor73-map-audit.json',PARENT/'native_source.v',parentmap,canonical)})
 (root/'service_clock_profile.json').write_text(json.dumps(result,indent=2));return result

def generate(root,reference):
 h,g,ps=pickle.loads((BOARD/'sensor73-map-parse.pkl').read_bytes());assert h==sha(root/'outflow/ti60f225_oob.map.v')
 def loc(p):return {v[0]:tuple(map(int,v[1:])) for line in p.read_text().splitlines() if len(v:=line.split())==4 and not line.startswith('#')}
 target=root/'fixed_dsp.placeloc';current=loc(target);original=loc(REFERENCE/'fixed_dsp.placeloc')
 assert len(current)==288 and len(original)==404
 assert all(original[n]==xyz for n,xyz in current.items())
 added=set(original)-set(current);assert len(added)==116
 assert {n for n in added if n.startswith('service_isp/')}=={'service_isp/red','service_isp/green','service_isp/blue'}
 assert all(g[n][0] in ('EFX_RAM10','EFX_DPRAM10') for n in added)
 assert sum(g[n][0]=='EFX_RAM10' for n in added)==32
 assert sum(g[n][0]=='EFX_DPRAM10' for n in added)==84
 assert all(g[n][0] in ('EFX_DSP24','EFX_DSP48') for n in current)
 shutil.copy2(REFERENCE/'fixed_dsp.placeloc',target)
 result=dict(status='RETAINED_RELEASED_HARD_ANCHORS_NO_ORDINARY_ANCHORS',candidate_map_sha256=h,
  assignments={n:list(x) for n,x in original.items()},total_hard_anchors=404,ordinary_anchors=0,
  assignments_sha256=sha(target),reference=str(REFERENCE),evidence={str(f):sha(f) for f in (Path(__file__),REFERENCE/'fixed_dsp.placeloc',BOARD/'sensor73-map-audit.json')},hardware_access=False)
 (root/'cnn_ram_placement.json').write_text(json.dumps(result,indent=2));return result

def verify_placed(root):
 proof=json.loads((root/'cnn_ram_placement.json').read_text())
 for f,h in proof['evidence'].items():assert sha(f)==h
 loc={v[0]:list(map(int,v[1:])) for line in (root/'outflow/ti60f225_oob.out.placeloc').read_text().splitlines() if len(v:=line.split())==4 and not line.startswith('#')}
 assert all(loc[n]==xyz for n,xyz in proof['assignments'].items())
 return dict(status='SENSOR73_404_HARD_ANCHORS_PLACED_PASS',count=404,ordinary_anchors=0,placement_sha256=sha(root/'outflow/ti60f225_oob.out.placeloc'))
