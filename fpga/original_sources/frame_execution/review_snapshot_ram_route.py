"""Inspect the actual RAM-snapshot placement before any route is started."""
import argparse,json
from pathlib import Path
from run_units import ROOT,HERE,sha
from snapshot_ram_board import BOARD,audit_delta
from snapshot_ram_cdc import structure
from audit_snapshot_cells import cells
import audit_control_state_implementation as implementation

PHYSICAL=ROOT/'tmp/frame-execution/snapshot-ram-place1'
EVIDENCE=ROOT/'tmp/frame-execution/snapshot-ram-evidence2'

def save(path,value):
 assert not path.exists(),path
 path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf8')

def prepare():
 audit_delta(BOARD);implementation.audit_delta=audit_delta
 save(PHYSICAL/'pre_route_implementation.json',implementation.audit(BOARD,PHYSICAL))
 save(PHYSICAL/'snapshot_ram_structure.json',structure(BOARD))
 cc=cells((BOARD/'outflow/ti60f225_oob.map.v').read_text())
 prefix='video_processing_pipeline/pipeline/cnn/core/dut/engine/'
 groups={}
 for suffix,width in (('rq/selected_feedback',288),('mac/source/back_payload',395),('mac/column_data',384),('od',768),('ox',27)):
  names=[prefix+suffix+f'[{i}]~FF' for i in range(width)]
  for n in names:assert cc[n][0]=='EFX_FF' and cc[n][1]['CLK']=='process_clk_150',n
  groups[suffix]=dict(count=width,enable_nets=sorted({cc[n][1]['CE'] for n in names}),
   reset_nets=sorted({cc[n][1]['SR'] for n in names}))
 save(PHYSICAL/'registered_boundary_map_structure.json',dict(groups=groups,map_sha256=sha(BOARD/'outflow/ti60f225_oob.map.v')))
 print('SAME_MAP_RESETS_PINS_RAM_AND_REGISTER_BOUNDARIES_PASS',flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--rationale')
 a=p.parse_args()
 if a.prepare_only:prepare();return
 assert a.rationale and len(a.rationale)>120,'Explicit review of measured tradeoffs required'
 audit_delta(BOARD)
 d=json.loads((PHYSICAL/'placement_review_inputs.json').read_text());c=d['current']
 assert c['placement_legal'] and c['optimization_setup_reserve_ns']==0
 assert all(v['used']<=v['capacity'] for v in c['resources'].values())
 assert c['packing']['COMB4']==c['packing']['COMB4 + FF']==0
 assert len(c['setup_clock_relationships'])==22 and c['canonical_setup_estimate_ns']>=0
 assert json.loads((PHYSICAL/'cnn_ram_actual_locations.json').read_text())['count']==113
 r=json.loads((EVIDENCE/'result.json').read_text())
 assert r['status']=='SNAPSHOT_RAM_SAME_MAP_SHORT_INTEGRATION_PASS'
 assert r['map_sha256']==sha(BOARD/'outflow/ti60f225_oob.map.v')
 for f,h in r['evidence'].items():assert sha(f)==h,f
 comparison={};paths=[PHYSICAL/x for x in ('placement_review_inputs.json','pre_route_implementation.json',
  'snapshot_ram_structure.json','registered_boundary_map_structure.json','cnn_ram_actual_locations.json',
  'physical_inputs.json','pnr_threads.json','result.json','outflow/ti60f225_oob.place.timing.rpt')]
 for ref in ('split-registered-place1','split-outputs-place3','startup-cause-place2'):
  old=ROOT/'tmp/frame-execution'/ref;record=old/'placement_review_inputs.json';paths.append(record)
  prev=json.loads(record.read_text())['current']
  for n in ('ti60f225_oob.sdc','fixed_dsp.placeloc','ti60f225_oob.peri.xml'):assert sha(PHYSICAL/n)==sha(old/n),(ref,n)
  comparison[ref]=dict(resources=prev['resources'],weighted_hpwl_pct=100*(c['weighted_hpwl_per_net']/prev['weighted_hpwl_per_net']-1),
   congestion_pct={k:100*(v/prev['congestion_peaks'][k]-1) for k,v in c['congestion_peaks'].items()})
 paths += [EVIDENCE/'result.json',EVIDENCE/'map-audit/result.json',EVIDENCE/'map-audit/ram_wrapper.json',
  EVIDENCE/'map-audit/snapshot_ram_structure.json',BOARD/'inputs.json',BOARD/'result.json',
  HERE/'route_snapshot_ram_board.py',HERE/'snapshot_ram_cdc.py',HERE/'audit_snapshot_ram_gray.py',Path(__file__)]
 review=dict(decision='PROCEED_TO_BOUNDED_ROUTE',physical=str(PHYSICAL),placement_review_sha256=sha(PHYSICAL/'placement_review_inputs.json'),
  rationale=a.rationale,comparisons=comparison,current=c,
  largest_fanouts=d['graph']['high_fanout_including_reset'][:16],
  remaining_risks=['One chosen implementation must route with zero conflicts and original WNS/WHS >=0.',
   'RAM payload CDC replaces the old wide data crossing; actual handshake and all vendor Gray checks remain required.',
   'Resources within absolute capacity are not a completed budget for all remaining baseline features.',
   'DDR calibration and board startup are still unverified; no download without user supervision.'],
  budget_seconds=1200,total_host_deadline_seconds=1800,route_attempts_authorized=1,
  required_evidence={str(f):sha(f) for f in paths},hardware_access=False,board_release=False)
 save(PHYSICAL/'route_review.json',review);print(json.dumps(comparison),flush=True)
if __name__=='__main__':main()
