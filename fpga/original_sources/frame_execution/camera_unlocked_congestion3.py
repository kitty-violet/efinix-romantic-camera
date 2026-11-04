"""One documented congestion recipe with all RCF locks removed, same placement."""
import inspect,json,sys
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from control_command_repair import once
import camera_checked_route_v3 as previous
import prepare_local_driver_route as copier
BASE=ROOT/'tmp/frame-execution';PHYSICAL=BASE/'camera-checked-bound10'

def prepare():
    # Do not start a second implementation while the frozen trial is running.
    failure=BASE/'camera-checked-route9/failure.json'
    assert failure.exists(), 'Wait for route9 terminal receipt'
    stages=json.loads((BASE/'camera-checked-route9/stages.json').read_text())
    assert stages and all(s['status']!='running' for s in stages)
    copy_code=once(inspect.getsource(copier.main),"'after':'TIMING_1'","'after':'CONGESTION_3'")
    env=dict(copier.__dict__,audit_delta=previous.camera_checked_board.audit_delta)
    exec(compile(copy_code,__file__+':copy-guard','exec'),env)
    prep=inspect.getsource(previous.prepare)
    prep=once(prep,'from prepare_local_driver_route import main as old_prepare','old_prepare=copy_prepare')
    prep=prep.replace('camera-checked-route-review3.json','camera-checked-route-review10.json')
    prep=once(prep,"    review_path.write_text(json.dumps(r,indent=2))", """    r['route_profile_cli_override']={'before':'CONGESTION_2','after':'CONGESTION_3'}
    first=json.loads((BASE/'camera-checked-route-review4b.json').read_text())
    r['rationale']=first['rationale'][:4]+[
        'RCF protection7/8/258 nets failed full setup; expanding to4551 complete-cone nets caused conflict reduction to stall and delay estimates to worsen. Do not grow the lock set or mix traces.',
        'Return to the same original reviewed placement and full map/SDC as the closest-0.045ns C2 route4. No sparse-route ini, RCF, extra cell anchors or traces are carried over. Preserve only original113RAM/288DSP anchors.',
        'Official2026.1 tim-pnr/cli-pnr-opts explicitly offer CONGESTION_1/2/3 for hard-to-route designs. Try CONGESTION_3 once instead of C2; keep seed3 and placement, not a seed sweep. C3 is a different congestion/timing allocation recipe, not a guarantee.',
        'This removes constrained-router limitations; the tool may use its normal parallel router subject to existing thread limits. Full original WNS/WHS>=0, coverage, CDC/Gray, reset/PLL and source gates remain. Tool1350s/host1395s. No hardware access.'
    ]
    r['remaining_risks']=['Different unconstrained allocation can still regress setup or hold. High utilization remains56637/60800XLR,228RAM,160DSP. No algorithmic or performance reduction is authorized.']
    for f in (failure,BASE/'camera-checked-route9/stages.json',BASE/'camera-checked-bound9/outflow/ti60f225_oob.route.out',
              BASE/'camera-checked-route4/selected.json',EF/'doc/topics/tim-pnr.html',EF/'doc/topics/cli-pnr-opts.html',EF/'doc/topics/rcf-bp.html',Path(__file__).resolve()):
        r['reviewed_evidence'][str(f)]=sha(f)
    review_path.write_text(json.dumps(r,indent=2))""")
    scope=dict(previous.__dict__,PHYSICAL=PHYSICAL,copy_prepare=env['main'],__file__=__file__,failure=failure,EF=EF)
    exec(compile(prep,__file__+':reviewed-prepare','exec'),scope);scope['prepare']()
    assert (PHYSICAL/'efx_pnr_settings.ini').read_text()=='loc_assignment = fixed_dsp.placeloc\ndump_placeloc = on\n'
    assert not list(PHYSICAL.glob('*.rcf')) and not list(PHYSICAL.glob('*.troutingtraces'))
    for n in ('outflow/ti60f225_oob.map.v','outflow/ti60f225_oob.place','outflow/ti60f225_oob.out.placeloc','ti60f225_oob.sdc'):
        assert sha(PHYSICAL/n)==sha(BASE/'camera-checked-bound4'/n),n
    print('UNCONSTRAINED_C3_SAME_MAP_PLACEMENT_CLOCKS_PREFLIGHT_PASS')

def route():
    from route_local_driver_board import driver_code
    from camera_checked_flow import generated
    source,code=driver_code();assert code.count('TIMING_1')==2
    code=code.replace('TIMING_1','CONGESTION_3')
    code=once(code,'from local_driver_board import audit_delta','from camera_checked_board import audit_delta')
    code=once(code,'import audit_control_state_implementation as impl','import audit_camera_implementation as impl')
    code=once(code,"HERE/'audit_native_split_gray.py'","HERE/'camera_checked_gray_v2.py'")
    generated(source,code,'-local-driver-route')
if __name__=='__main__':
    action=sys.argv.pop(1)
    if action=='prepare':prepare()
    elif action=='route':route()
    else:raise ValueError(action)
