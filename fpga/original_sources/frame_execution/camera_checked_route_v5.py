"""Same C2 placement/router, one measured path gets a tighter routing target."""
import inspect,json,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_command_repair import once
import camera_checked_route_v4 as previous

BASE=ROOT/'tmp/frame-execution';PHYSICAL=BASE/'camera-checked-bound5'

def prepare():
    code=inspect.getsource(previous.prepare)
    code=once(code,'camera-checked-route-review4b.json','camera-checked-route-review5.json')
    scope=dict(previous.__dict__,PHYSICAL=PHYSICAL,__file__=__file__)
    exec(compile(code,__file__+':prepare','exec'),scope);scope['prepare']()
    from camera_targeted_route_margin import apply
    apply(PHYSICAL)
    f=PHYSICAL/'route_review.json';r=json.loads(f.read_text())
    r['rationale'] += [
        'C2 route4 completed58iterations in559.922seconds. Canonical WNS-0.045ns/WHS+0.022ns; only one reported setup failure, held_owner4/1 through two LUTs to arenaROW4HALF1SLICE2 WDATAB0. All remaining reported critical paths positive.',
        'Keep C2, seed3, exact map/placement and all physical clocks. Add only a6.450ns max_delay routing target on that real path. Frozen-route probe confirms its slack tightens from-0.045 to-0.261ns. No false_path, multicycle or relaxed clock constraint.',
        'Final canonical STA removes exactly this additional target and binds the original acceptance SDC. Release still requires every original setup/hold check, full actual CDC/Gray/PLL and source binding.'
    ]
    for p in (BASE/'camera-checked-route4/selected.json',BASE/'camera-checked-route4/failure.json',
              HERE/'camera_checked_route_v4.py',Path(__file__).resolve()):r['required_evidence'][str(p)]=sha(p)
    f.write_text(json.dumps(r,indent=2));print('ONE_PATH_TIGHTENED_C2_ROUTE_READY')

def route():
    from route_local_driver_board import driver_code
    from camera_checked_flow import generated
    source,code=driver_code()
    code=once(code,'from local_driver_board import audit_delta','from camera_checked_board import audit_delta')
    code=once(code,'import audit_control_state_implementation as impl','import audit_camera_implementation as impl')
    code=once(code,"HERE/'audit_native_split_gray.py'","HERE/'camera_checked_gray_v2.py'")
    code=once(code,"HERE/'canonical_snapshot_sta.py','--base'","HERE/'camera_targeted_route_margin.py','canonical','--base'")
    assert code.count('TIMING_1')==2;code=code.replace('TIMING_1','CONGESTION_2')
    generated(source,code,'-local-driver-route')

if __name__=='__main__':
    action=sys.argv.pop(1)
    if action=='prepare':prepare()
    elif action=='route':route()
    else:raise ValueError(action)
