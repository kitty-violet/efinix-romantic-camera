"""Congestion-aware routing recipe on the same proven camera map/placement.

The first corrected route reduced setup failure to0.205ns but several unrelated
CNN/RAM/DSP cones remain wire limited. Change only the official router recipe,
not seed, coordinates, logic, clock model or acceptance requirements.
"""
import inspect,json,sys
from run_units import ROOT,HERE,sha
from control_command_repair import once
import camera_checked_route_v3 as previous
import prepare_local_driver_route as copier

BASE=ROOT/'tmp/frame-execution'
PHYSICAL=BASE/'camera-checked-bound4'

def prepare():
    # This is the original copy-and-bind implementation, with only its recipe
    # whitelist changed from TIMING_1 to CONGESTION_2. Bind both source bodies.
    copy_code=inspect.getsource(copier.main)
    copy_code=once(copy_code,"'after':'TIMING_1'","'after':'CONGESTION_2'")
    env=dict(copier.__dict__,audit_delta=previous.camera_checked_board.audit_delta)
    exec(compile(copy_code,__file__+':copy-guard','exec'),env)
    prep_code=inspect.getsource(previous.prepare)
    prep_code=once(prep_code,'from prepare_local_driver_route import main as old_prepare','old_prepare=copy_prepare')
    prep_code=prep_code.replace('camera-checked-route-review3.json','camera-checked-route-review4b.json')
    needle='    review_path.write_text(json.dumps(r,indent=2))'
    added="""    r['route_profile_cli_override']={'before':'CONGESTION_2','after':'CONGESTION_2'}
    r['rationale']=r['rationale'][:3]+[
        'All same-map camera and control short tests remain frozen and valid. Clock model is exactly the reviewed config-only correction; physical150MHz/DDR/video clocks unchanged.',
        'Corrected TIMING_1 route3 is legal in71iterations,728.438seconds, but setup is-0.205ns. Its worst paths span requantization DSP output, MAC sums, row enables and RAM reads. They are wire limited rather than one incorrect camera net.',
        'Failed route2 full-fanout survey shows the two original hot nets have smaller bounding boxes than the passing parent, so no evidence supports simply moving those drivers closer. Keep ordinary-cell placement frozen for this controlled routing comparison.',
        'Use official CONGESTION_2 router recipe once on existing reviewed CONGESTION_2 placement. The vendor guide distinguishes congestion-oriented recipes from TIMING recipes; >70iterations and wire-dominated residual paths justify this change. Recipe selection is not a guarantee and no seed sweep is launched.',
        'Preserve all failed receipts. Tool1350seconds and independent host1395seconds; all strict setup/hold, actual CDC/Gray, PLL and final source binding still required before board readiness.'
    ]
    for f in (BASE/'camera-checked-route3/selected.json',BASE/'camera-checked-route3/failure.json',
              BASE/'camera-checked-bound3/outflow/ti60f225_oob.timing.rpt',
              BASE/'camera-checked-route2/hotnet_geometry_comparison.json',Path(__file__).resolve()):
        r['reviewed_evidence'][str(f)]=sha(f)
    r['copy_profile_guard_adapter']=dict(source_sha256=sha(HERE/'prepare_local_driver_route.py'),
        adapter_sha256=sha(__file__),change='Only accepted routing recipe in copy precondition; all source/capacity/test/geometry bindings preserved')
    review_path.write_text(json.dumps(r,indent=2))"""
    prep_code=once(prep_code,needle,added)
    scope=dict(previous.__dict__,PHYSICAL=PHYSICAL,copy_prepare=env['main'],__file__=__file__)
    from pathlib import Path
    scope['Path']=Path
    exec(compile(prep_code,__file__+':prepare','exec'),scope);scope['prepare']()

def route():
    code=inspect.getsource(previous.route)
    code=once(code,'    source,code=driver_code()',
        "    source,code=driver_code()\n    assert code.count('TIMING_1')==2\n    code=code.replace('TIMING_1','CONGESTION_2')")
    scope=dict(previous.__dict__)
    exec(compile(code,__file__+':route','exec'),scope);scope['route']()

if __name__=='__main__':
    action=sys.argv.pop(1)
    if action=='prepare':prepare()
    elif action=='route':route()
    else:raise ValueError(action)
