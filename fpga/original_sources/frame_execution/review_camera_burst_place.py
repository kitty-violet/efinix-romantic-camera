"""Inspect complete placement and capture-specific loads before any route."""
import argparse,collections,json
from pathlib import Path
from run_units import ROOT,sha
from camera_capture_burst_board import BOARD,PARENT,RESET
from camera_ratio_before_place import metrics
from native_local_drivers import graph
from fixed_control_ddr_reuse_locations import locations
def main():
 p=argparse.ArgumentParser();p.add_argument('--placement',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();place=a.placement.resolve();assert not a.output.exists()
 r=json.loads((place/'placement_review_inputs.json').read_text());assert r['status']=='PLACED_STOPPED_FOR_ROUTABILITY_REVIEW'
 ref=metrics(ROOT/'tmp/frame-execution/camera-recovery-place1');current=r['current']
 old=graph((PARENT/'outflow/ti60f225_oob.map.v').read_text());new=graph((BOARD/'outflow/ti60f225_oob.map.v').read_text());xy=locations(place/'outflow/ti60f225_oob.out.placeloc')
 output={'EFX_LUT4':['O'],'EFX_FF':['Q'],'EFX_ADD':['O','CO'],'EFX_RAM10':['RDATA'],'EFX_DPRAM10':['RDATA'],'EFX_SRL8':['Q']}
 def loads(g):
  c=collections.defaultdict(set)
  for name,(kind,pins) in g.items():
   for pin,ts in pins.items():
    if pin not in output.get(kind,[]):
     for t in ts:c[t].add((name,pin))
  return c
 before,after=loads(old),loads(new)
 def box(names):
  pts=[xy[n] for n in names if n in xy]
  return [min(v[0] for v in pts),min(v[1] for v in pts),max(v[0] for v in pts),max(v[1] for v in pts)] if pts else None
 def row(name,category):
  token=('net',name,None);sinks={n for n,p in after[token]}
  return dict(net=name,category=category,before=len({n for n,p in before[token]}),after=len(sinks),pins=sorted({p for n,p in after[token]}),bbox=box(sinks))
 names=[('process_clk_150','clock'),('i_mipi_rx_pclk','clock'),(RESET,'capture_reset'),('w_arstn','system_reset'),('cpu/system_hardJtag_debug_logic_jtagBridge/observation/request_sync','snapshot_select'),('cap_ready','capture_backpressure'),('cap_active','capture_state'),('cap_awvalid','write_address'),('cap_wvalid','write_data'),('cap_wlast','write_data'),('video_processing_pipeline/down_valid','capture_input'),('control_cancel','cancel'),('raw_capture_abort','abort')]
 additions=json.loads((BOARD/'result.json').read_text())['added_cells'];new_rams={n:xy.get(n) for n in additions if new[n][0] in ('EFX_RAM10','EFX_DPRAM10')}
 assert len(new_rams)==7 and all(v is not None for v in new_rams.values()),new_rams
 files=[place/n for n in ('physical_inputs.json','result.json','placement_review_inputs.json','cnn_ram_actual_locations.json','cnn_ram_placement.json','fixed_dsp.placeloc','outflow/ti60f225_oob.out.placeloc')]
 files += [BOARD/'result.json',ROOT/'tmp/frame-execution/camera-capture-integrated-cut2/result.json',ROOT/'tmp/frame-execution/camera-burst-evidence1/result.json',ROOT/'tmp/board-test/camera-burst-progress1.json',ROOT/'tmp/board-test/camera-burst-cdc-structure1.json',Path(__file__)]
 report=dict(status='CAPTURE_BURST_PLACEMENT_REVIEW_NOT_ROUTE_APPROVAL',current=current,reference=ref,
  changes_pct=dict(hpwl=100*(current['weighted_hpwl_per_net']/ref['weighted_hpwl_per_net']-1),**{k:100*(current['congestion_peaks'][k]/v-1) for k,v in ref['congestion_peaks'].items()}),
  affected_loads=[row(*x) for x in names],new_capture_ram_locations=new_rams,capture_logic_bbox=box(additions),added_primitives=dict(collections.Counter(new[n][0] for n in additions)),global_graph=r['graph'],evidence={str(f):sha(f) for f in files},hardware_access=False,board_release=False)
 a.output.write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ('status','changes_pct','affected_loads','new_capture_ram_locations')}));print(json.dumps({k:current[k] for k in ('resources','congestion_peaks','weighted_hpwl_per_net','canonical_setup_estimate_ns')}))
if __name__=='__main__':main()
