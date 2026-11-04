"""Whole-layout and changed-control review, never an automatic route approval."""
import argparse,collections,json
from pathlib import Path
from run_units import ROOT,HERE,sha
from camera_ready_board import BOARD,PARENT
from camera_ratio_before_place import metrics
from native_local_drivers import graph
from fixed_control_ddr_reuse_locations import locations
def main():
 p=argparse.ArgumentParser();p.add_argument('--placement',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();place=a.placement.resolve();out=a.output.resolve();assert not out.exists()
 terminal=json.loads((place/'result.json').read_text());assert terminal['status']=='PLACEMENT_ONLY_FINISHED_NOT_RELEASE' and all(s['exit_code']==0 for s in terminal['stages'])
 review=json.loads((place/'placement_review_inputs.json').read_text());current=review['current']
 refs={n:metrics(ROOT/'tmp/frame-execution'/n) for n in ('camera-payload-place1','camera-checked-place2','camera-checked-place3','native-local-driver-place1')}
 changes={n:dict(hpwl_pct=100*(current['weighted_hpwl_per_net']/r['weighted_hpwl_per_net']-1),**{k:100*(current['congestion_peaks'][k]/v-1) for k,v in r['congestion_peaks'].items()}) for n,r in refs.items()}
 g=graph((BOARD/'native_source.v').read_text());old=graph((PARENT/'outflow/ti60f225_oob.map.v').read_text());xy=locations(place/'outflow/ti60f225_oob.out.placeloc')
 def loads(g):
  c=collections.defaultdict(set)
  for n,(k,pins) in g.items():
   o={'EFX_LUT4':'O','EFX_FF':'Q','EFX_ADD':'O','EFX_SRL8':'Q'}.get(k)
   for p,ts in pins.items():
    if p!=o:
     for t in ts:c[t].add(n)
  return c
 before,after=loads(old),loads(g);defs=json.loads((ROOT/'tmp/frame-execution/camera-ready-native4/definitions.json').read_text());controls=[]
 for n in defs:
  outtoken=g[n][1]['O'][0];sinks=after[outtoken]
  controls.append(dict(cell=n,location=xy[n],before_loads=len(before[outtoken]),after_loads=len(sinks),max_manhattan=max([abs(xy[n][0]-xy[k][0])+abs(xy[n][1]-xy[k][1]) for k in sinks if k in xy] or [0])))
 assert all(k!='EFX_FF' or (n in old and (k,pins)==old[n]) for n,(k,pins) in g.items())
 files=[place/'physical_inputs.json',place/'result.json',place/'placement_review_inputs.json',place/'cnn_ram_actual_locations.json',place/'cnn_ram_placement.json',place/'fixed_dsp.placeloc',place/'outflow/ti60f225_oob.out.placeloc',BOARD/'result.json',ROOT/'tmp/frame-execution/camera-ready-evidence1/result.json',ROOT/'tmp/board-test/camera-ready-release-preflight1.json',Path(__file__)]
 result=dict(status='READY_PLACEMENT_REVIEW_DATA_NOT_ROUTE_APPROVAL',current=current,references=refs,changes_pct=changes,changed_control_outputs=controls,ff_clock_enable_reset_unchanged=True,graph=review['graph'],evidence={str(f):sha(f) for f in files},hardware_access=False,board_release=False)
 out.write_text(json.dumps(result,indent=2));print(json.dumps(dict(changes_pct=changes,current=current,changed_control_outputs=controls)))
if __name__=='__main__':main()
