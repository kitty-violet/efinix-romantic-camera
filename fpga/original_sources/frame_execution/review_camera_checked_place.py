"""Finish only placement reporting; never rerun successful placement."""
import inspect,json
from run_units import ROOT,HERE,sha
import audit_global_place as old
from review_existing_routability import graph_review
from fixed_cnn_ram_locations import verify_placed
from camera_checked_board import BOARD,audit_delta

def main():
 root=ROOT/'tmp/frame-execution/camera-checked-place1';ref=ROOT/'tmp/frame-execution/native-local-driver-place1'
 assert not (root/'placement_review_inputs.json').exists();audit_delta(BOARD)
 result=json.loads((root/'result.json').read_text());assert result['status']=='PLACEMENT_ONLY_FINISHED_NOT_RELEASE'
 assert all(x['exit_code']==0 for x in result['stages']) and not (root/'outflow/ti60f225_oob.troutingtraces').exists()
 code=inspect.getsource(old.metrics).replace('assert len(rows)==22,len(rows)','assert 22<=len(rows)<=24,len(rows)')
 scope=dict(old.__dict__);exec(compile(code,__file__+':metrics','exec'),scope)
 current=scope['metrics'](root);previous=old.metrics(ref)
 clocks=lambda x,k:{(r['launch'],r['capture'],r['edge']) for r in x[k]}
 for key in ('setup_clock_relationships','hold_clock_relationships'):
  added=clocks(current,key)-clocks(previous,key);lost=clocks(previous,key)-clocks(current,key)
  assert not lost and added<= {('CLK_5M','process_clk_150','R-R'),('i_mipi_rx_pclk','process_clk_150','R-R')},(added,lost)
  print(key,'new',sorted(added),flush=True)
 assert verify_placed(root)==json.loads((root/'cnn_ram_actual_locations.json').read_text())
 review=dict(status='PLACED_STOPPED_FOR_ROUTABILITY_REVIEW',current=current,reference=previous,
  changes_pct=dict(weighted_hpwl=100*(current['weighted_hpwl_per_net']/previous['weighted_hpwl_per_net']-1),
  **{k:100*(current['congestion_peaks'][k]/v-1) for k,v in previous['congestion_peaks'].items()}),
  graph=graph_review(root),reference_path=str(ref),route_started=False,hardware_access=False,board_release=False,
  reporting_repair='Original 22-clock-row parser rejected the actual added camera crossing. Reports only regenerated; geometry untouched.',auditor_sha256=sha(__file__))
 (root/'placement_review_inputs.json').write_text(json.dumps(review,indent=2))
 print(json.dumps({k:review[k] for k in ('changes_pct',)}));print(json.dumps({k:v for k,v in current.items() if k not in ('setup_clock_relationships','hold_clock_relationships','evidence')}))
if __name__=='__main__':main()
