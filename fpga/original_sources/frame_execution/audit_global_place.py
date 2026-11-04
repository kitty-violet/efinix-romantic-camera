"""Capacity, packing, all-clock placement estimates and whole-chip congestion."""
import json,re
from pathlib import Path
from run_units import ROOT,sha

REFERENCE=ROOT/'tmp/frame-execution/snapshot-physical2/pnr-seed3'

def metrics(root):
    out=root/'outflow';pl=out/'ti60f225_oob.place.out';rpt=out/'ti60f225_oob.place.rpt'
    body=pl.read_text();resource=rpt.read_text()
    resources={k:dict(used=int(a),capacity=int(b)) for k,a,b in
      re.findall(r'^(XLRs|Memory Blocks|DSP Blocks):\s*(\d+)\s*/\s*(\d+)',resource,re.M)}
    assert len(resources)==3
    packing={k:int(v) for k,v in re.findall(r'XLRs needed for ([^:]+):\s*(\d+)',resource)}
    peaks={k:float(v) for k,v in re.findall(r'Peak congestion smeared over (4x4|1/16|1/4) of the chip is ([\d.]+)',body)}
    assert len(peaks)==3
    hpwl=float(re.search(r'Congestion-weighted HPWL per net: ([\d.]+)',body)[1])
    setup=body.split('Setup (Max) Clock Relationship',1)[1].split('Hold (Min) Clock Relationship',1)[0]
    hold=body.split('Hold (Min) Clock Relationship',1)[1]
    def clockrows(s):
        rows=[dict(launch=a,capture=b,constraint_ns=float(c),slack_ns=float(v),edge=e) for a,b,c,v,e in
         re.findall(r'^\s*(\w+)\s+(\w+)\s+([\d.-]+)\s+([\d.-]+)\s+\(([^)]+)\)',s,re.M)]
        assert len(rows)==22,len(rows);return rows
    checks=all(x in body for x in ('Post-placement cluster-level checks is successful',
                                  'Post-placement primitive-level checks is successful'))
    meta=json.loads((root/'physical_inputs.json').read_text())
    reserve=(meta.get('setup_margin') or {}).get('extra_setup_ns',0)
    setup_rows=clockrows(setup)
    return dict(resources=resources,packing=packing,congestion_peaks=peaks,weighted_hpwl_per_net=hpwl,
                optimization_setup_reserve_ns=reserve,
                canonical_setup_estimate_ns=min(r['slack_ns'] for r in setup_rows)+reserve,
                setup_clock_relationships=setup_rows,hold_clock_relationships=clockrows(hold),
                placement_legal=checks,evidence={str(pl):sha(pl),str(rpt):sha(rpt)})

def audit(root,reference_root=REFERENCE):
    current=metrics(root);reference=metrics(reference_root);reasons=[]
    for key,r in current['resources'].items():
        if r['used']>r['capacity']:reasons.append(key+' over capacity')
    if not current['placement_legal']:reasons.append('placement legal checks missing')
    # Reject clear regression using the old passing board, not invented absolute
    # congestion limits. These are experiment screening thresholds, not STA.
    for k,v in current['congestion_peaks'].items():
        if v>reference['congestion_peaks'][k]*1.10:reasons.append('congestion '+k+' over reference +10%')
    if current['weighted_hpwl_per_net']>reference['weighted_hpwl_per_net']*1.05:
        reasons.append('weighted wirelength over reference +5%')
    if current['canonical_setup_estimate_ns']<0:
        reasons.append('negative original-constraint placement setup estimate')
    return dict(status='PLACEMENT_SCREEN_PASS_ROUTED_STA_REQUIRED' if not reasons else 'PLACEMENT_SCREEN_REJECT',
        reasons=reasons,current=current,reference=reference,reference_root=str(reference_root),
        limits=['Placement estimates do not establish final routed timing or CDC.',
                'Negative pre-route hold estimates require router hold repair and final nonnegative WHS.',
                'The deferred extra setup reserve is optimization-only; screening adds it back to estimate the original constraint.',
                'The 10% congestion and 5% wirelength bounds are screening heuristics, not vendor signoff.',
                'Routing permission additionally requires mapped fanout and source/structure checks.'],
        hardware_access=False,board_release=False)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    print(json.dumps(audit(a.root.resolve()),indent=2))
