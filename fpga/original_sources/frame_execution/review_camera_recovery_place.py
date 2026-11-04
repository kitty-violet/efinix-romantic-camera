"""Whole-layout plus all added camera clock/reset/data loads; no routing."""
import argparse,collections,json
from pathlib import Path
from run_units import ROOT,sha
from camera_recovery_board import BOARD,PARENT
from camera_ratio_before_place import metrics
from native_local_drivers import graph
from fixed_control_ddr_reuse_locations import locations

def main():
 p=argparse.ArgumentParser();p.add_argument('--placement',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();place=a.placement.resolve();assert not a.output.exists()
 r=json.loads((place/'placement_review_inputs.json').read_text());assert r['status']=='PLACED_STOPPED_FOR_ROUTABILITY_REVIEW'
 current=r['current'];ref=metrics(ROOT/'tmp/frame-execution/camera-ready-place1')
 old=graph((PARENT/'outflow/ti60f225_oob.map.v').read_text());new=graph((BOARD/'outflow/ti60f225_oob.map.v').read_text());xy=locations(place/'outflow/ti60f225_oob.out.placeloc')
 def loads(g):
  c=collections.defaultdict(set)
  for n,(k,p) in g.items():
   for pin,ts in p.items():
    if pin not in {'EFX_LUT4':['O'],'EFX_FF':['Q'],'EFX_ADD':['O','CO'],'EFX_RAM10':['RDATA'],'EFX_DPRAM10':['RDATA'],'EFX_SRL8':['Q']}.get(k,[]):
     for t in ts:c[t].add((n,pin))
  return c
 before,after=loads(old),loads(new);affected=[]
 def row(token,category):
  sinks={n for n,p in after[token]};pts=[xy[n] for n in sinks if n in xy]
  return dict(token=token,category=category,before=len({n for n,p in before[token]}),after=len(sinks),
   pins=sorted({p for n,p in after[token]}),bbox=[min(v[0] for v in pts),min(v[1] for v in pts),max(v[0] for v in pts),max(v[1] for v in pts)] if pts else None)
 for name,category in [('i_mipi_rx_pclk','clock'),('process_clk_150','clock'),('raw_frontend/ingress/n10','pixel_release'),('w_arstn','system_reset'),('cpu/system_hardJtag_debug_logic_jtagBridge/observation/request_sync','system_page_select'),('camera_frame_start','raw_control'),('camera_frame_end','raw_control'),('camera_pixel_valid','raw_control'),('camera_frame_active','state'),('camera_boundary_error','sticky_fault'),('camera_polarity_known','state'),('camera_active_vsync_level','state')]:affected.append(row(('net',name,None),category))
 for i in range(19):affected.append(row(('net','raw_frontend/ingress/packer/captured_words',i),'read_only_count_tap'))
 affected.append(row(('net','raw_stream_errors',7),'read_only_fault_tap'))
 additions=json.loads((BOARD/'result.json').read_text())['added_cells'];counts=collections.Counter(new[n][0] for n in additions)
 files=[place/n for n in ('physical_inputs.json','result.json','placement_review_inputs.json','cnn_ram_actual_locations.json','cnn_ram_placement.json','fixed_dsp.placeloc','outflow/ti60f225_oob.out.placeloc')]
 files += [BOARD/'result.json',ROOT/'tmp/frame-execution/camera-recovery-mapped2/result.json',ROOT/'tmp/frame-execution/camera-recovery-adjacent1/result.json',ROOT/'tmp/board-test/camera-recovery-cdc-structure1.json',Path(__file__)]
 report=dict(status='RECOVERY_PLACEMENT_REVIEW_NOT_ROUTE_APPROVAL',current=current,reference=ref,
  changes_pct=dict(hpwl=100*(current['weighted_hpwl_per_net']/ref['weighted_hpwl_per_net']-1),**{k:100*(current['congestion_peaks'][k]/v-1) for k,v in ref['congestion_peaks'].items()}),
  affected_loads=affected,added_primitives=counts,global_graph=r['graph'],evidence={str(f):sha(f) for f in files},hardware_access=False,board_release=False)
 a.output.write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ('status','changes_pct','affected_loads','added_primitives')}));print(json.dumps({k:current[k] for k in ('resources','congestion_peaks','weighted_hpwl_per_net','canonical_setup_estimate_ns')}))
if __name__=='__main__':main()
