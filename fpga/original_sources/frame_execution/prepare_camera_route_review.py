"""Explicit current-placement review before one bounded route."""
import json
from run_units import ROOT,HERE,sha
from camera_checked_board import BOARD,audit_delta
from finish_camera_checked_release import camera_evidence

def main():
 base=ROOT/'tmp/frame-execution';place=base/'camera-checked-place1';shorts=base/'camera-checked-evidence1'
 out=base/'camera-checked-route-review1.json';assert not out.exists();audit_delta(BOARD)
 evidence={}
 def bind(f):evidence[str(f)]=sha(f);return json.loads(f.read_text())
 def verify(rows):
  for f,h in rows.items():assert sha(f)==h;evidence[str(f)]=h
 camera_evidence(bind,verify)
 sr=bind(shorts/'result.json');assert sr['status']=='LOCAL_DRIVER_SAME_MAP_SHORT_INTEGRATION_PASS';verify(sr['evidence'])
 r=bind(place/'placement_review_inputs.json');c=r['current']
 assert c['placement_legal'] and c['canonical_setup_estimate_ns']>=0
 assert all(v['used']<=v['capacity'] for v in c['resources'].values())
 for k in ('current','reference'):verify(r[k]['evidence'])
 verify(r['graph']['evidence'])
 for f in (place/'result.json',place/'physical_inputs.json',place/'cnn_ram_actual_locations.json',BOARD/'result.json',HERE/'review_camera_checked_place.py',HERE/'prepare_camera_route_review.py'):bind(f) if f.suffix=='.json' else verify({str(f):sha(f)})
 review=dict(decision='PROCEED_TO_BOUNDED_ROUTE',board=str(BOARD),placement=str(place),shorts=str(shorts),physical=str(base/'camera-checked-bound1'),budget_seconds=1350,
  route_profile_cli_override=dict(before='CONGESTION_2',after='TIMING_1'),
  rationale=[
   'Full camera fix adds292 XLR over selected parent, total56924/60800; RAM228/256 and DSP160/160 unchanged. No frequency/pin/timing exception change.',
   'Against the actual passing parent placement: fine4x4 congestion -0.78%, medium -4.89%; coarse +1.46% and weighted HPWL +2.09%. Added routing-only46 cells. This modest whole-chip tradeoff has improved fine/medium hotspots, not only a local distance claim.',
   'All23 setup relationships have positive placement estimates (minimum0.382ns). Process150 estimate0.568ns. The added 5MHz->process crossing is explicitly retained and will receive routed timing plus95-bit held-data checks.',
   'No ordinary logic fixed;401 existing113 CNN RAM/288 DSP anchors verified. Mapped global clock/reset/high-fanout nets remain included in graph review,74 high-fanout nets total; observations reuse existing111 serial held FFs.',
   'Actual full-board control/adapter/reset/serial current-map tests pass. Camera10 RTLcases and mapped0/4/6/8 tests pass;117 host bit checks and mapped16 one-shot enable checks pass. Existing protected interface and all noncamera connectivity preserved.'
  ],remaining_risks=[
   'Pre-route negative hold estimates require legal router hold repair. Setup estimates do not establish routed WNS; whole placement movement may expose other slow paths.',
   'Original Gray endpoint names require exact binding to physical cell names before route; constraints values and relationships remain unchanged.',
   'Must finish zero-overlap route, original WNS/WHS>=0, resource capacity,95-bit camera held settling,70-bit DDR Gray and JTAG CDC/PLL/bitstream binding before packaging.',
   'Physical camera ID/module reset polarity and actual input lines remain board tests; no automatic download is authorized.'
  ],reviewed_evidence=evidence,hardware_access=False,board_release=False)
 out.write_text(json.dumps(review,indent=2));print(out)
if __name__=='__main__':main()
