"""Compare whole-chip estimates and the added local FF control load."""
import argparse,collections,json
from pathlib import Path
from run_units import ROOT,HERE,sha
from camera_payload_board import BOARD,PARENT
from camera_ratio_before_place import metrics
from native_local_drivers import graph
from fixed_control_ddr_reuse_locations import locations

def main():
    p=argparse.ArgumentParser();p.add_argument('--placement',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    place=a.placement.resolve();out=a.output.resolve();assert not out.exists()
    terminal=json.loads((place/'result.json').read_text());assert terminal['status']=='PLACEMENT_ONLY_FINISHED_NOT_RELEASE'
    assert all(s['exit_code']==0 for s in terminal['stages'])
    review=json.loads((place/'placement_review_inputs.json').read_text());current=review['current']
    refs={n:metrics(ROOT/'tmp/frame-execution'/n) for n in ('camera-checked-place2','camera-checked-place3','native-local-driver-place1')}
    changes={n:dict(hpwl_pct=100*(current['weighted_hpwl_per_net']/r['weighted_hpwl_per_net']-1),
         **{k:100*(current['congestion_peaks'][k]/v-1) for k,v in r['congestion_peaks'].items()}) for n,r in refs.items()}
    old=graph((PARENT/'outflow/ti60f225_oob.map.v').read_text());new=graph((BOARD/'native_source.v').read_text())
    def loads(g):
        counts=collections.Counter()
        for n,(k,p) in g.items():
            for pin,ts in p.items():
                if (k,pin) in (('EFX_FF','Q'),('EFX_LUT4','O'),('EFX_ADD','O'),('EFX_ADD','CO'),('EFX_SRL8','Q')):continue
                for t in ts:counts[t]+=1
        return counts
    before,after=loads(old),loads(new)
    groups=json.loads((ROOT/'tmp/frame-execution/camera-payload-native2/groups.json').read_text())
    control={}
    for r in groups:
        for p in ('D','CE','SR','CLK'):
            t=new[r['new_cell']][1][p][0];key=str((p,t))
            control[key]=dict(pin=p,signal=t,before=before[t],after=after[t],increase=after[t]-before[t])
    xy=locations(place/'outflow/ti60f225_oob.out.placeloc')
    localized=[]
    for r in groups:
        x=xy[r['new_cell']];pts=[xy[l['cell']] for l in r['loads']]
        localized.append(dict(ff=r['new_cell'],bank=r['bank'],fanout=len(pts),location=x,
            bbox=[min(t[d] for t in pts) for d in (0,1)]+[max(t[d] for t in pts) for d in (0,1)],
            max_manhattan=max(abs(x[0]-t[0])+abs(x[1]-t[1]) for t in pts)))
    files=[place/'physical_inputs.json',place/'result.json',place/'placement_review_inputs.json',place/'cnn_ram_actual_locations.json',
      place/'fixed_dsp.placeloc',place/'outflow/ti60f225_oob.out.placeloc',BOARD/'result.json',
      ROOT/'tmp/frame-execution/camera-payload-evidence1/result.json',ROOT/'tmp/board-test/camera-payload-release-preflight1.json',Path(__file__)]
    result=dict(status='PAYLOAD_PLACEMENT_REVIEW_DATA_NOT_ROUTE_APPROVAL',current=current,references=refs,changes_pct=changes,
      control_load_changes=list(control.values()),local_drivers=localized,graph=review['graph'],
      evidence={str(f):sha(f) for f in files},hardware_access=False,board_release=False)
    out.write_text(json.dumps(result,indent=2));print(json.dumps(dict(changes_pct=changes,
      current=current,control_load_changes=list(control.values())),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
