"""Record the explicit third-placement review, not an automatic threshold gate."""
import json
from run_units import ROOT,HERE,sha
from audit_grouped_dsp_locations import audit

p=ROOT/'tmp/frame-execution/local-decode-quadrant-place3'
out=p/'route_review.json'
assert not out.exists()
assert audit(p)['status']=='NATIVE_288_DSP_PRIMITIVES_160_SITES_MATCH_GROUP_PLAN'
m=json.loads((p/'placement_review_inputs.json').read_text())
assert m['current']['resources']['XLRs']['used']==55901
assert m['current']['weighted_hpwl_per_net']==67.9733
assert m['current']['congestion_peaks']=={'4x4':1.82879,'1/16':1.55949,'1/4':1.31531}
assert m['current']['canonical_setup_estimate_ns']==.365
old=json.loads((ROOT/'tmp/frame-execution/local-decode-c1-place2/route_review.json').read_text())
files=[path for path in old['required_evidence'] if 'local-decode-c1-place2' not in path and 'probe_local_decode_place_v2.py' not in path]
files += [p/n for n in ('placement_review_inputs.json','physical_inputs.json','cnn_ram_actual_locations.json',
    'critical_geometry_review.json','grouped_dsp_actual_locations.json','quadrant_dsp_plan.json')]
files += [HERE/n for n in ('probe_local_decode_quadrant_place.py','fixed_quadrant_dsp_locations.py',
    'audit_grouped_dsp_locations.py','inspect_routed_failure_geometry.py','prepare_quadrant_route_review.py')]
for path in files:assert __import__('pathlib').Path(path).is_file(),path
result=dict(decision='PROCEED_TO_BOUNDED_ROUTE',physical=str(p),placement_review_sha256=sha(p/'placement_review_inputs.json'),
    rationale='Same map/RTL/SDC and passing affected short tests. Actual complete DSP groups and retained RAM anchors audited. Versus first failed C2: weighted HPWL -3.141%, congestion peaks -10.646/-9.860/-3.093%; all four also improve versus the prior passing implementation. Compared with C1, HPWL and 1/16 improve, 4x4/1/4 worsen slightly. All 22 setup estimates positive; compute +0.652ns, global +0.365ns. The most recent worst path contracts 99 to 20 tiles, its 7.783ns net 57 to 12; shared DSP groups are genuinely more compact. This offers a different physical basis from a recipe change. One bounded route measures real outcome, without claiming geometric distance predicts delay.',
    remaining_risks=['Only 3/10 first-failure and 1/10 second-failure total path distances improve. Several queue/RAM paths lengthen; full timing remains mandatory.',
        'Second-failure history BIT29 to DSP path lengthens 57 to 109; weighted congestion estimates cannot rule out detours.',
        'High fanout queue/control nets and pre-route hold repair remain. No new frequency, timing exception or PHY change.',
        'Controller startup and DDR calibration on the board remain unresolved; this is not a board release.'],
    budget_seconds=1200,route_attempts_authorized=1,source_and_constraints_unchanged=True,
    required_evidence={str(__import__('pathlib').Path(path)):sha(path) for path in files},hardware_access=False,board_release=False)
out.write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(dict(decision=result['decision'],evidence_count=len(files),review_sha256=sha(out))))
