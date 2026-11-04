"""Preserve complete measured combinational fan-in, including reviewed CE loads."""
import sys
from run_units import HERE,sha
from control_command_repair import once
import camera_sparse_route_v3 as previous
from camera_route_cone_plan_v2 import build

body=previous.body.replace('camera-checked-bound8','camera-checked-bound9').replace('camera-checked-route-review8','camera-checked-route-review9')
body=body.replace('MEASURED_DATA_CONES_AND_ONE_ENABLE_PREPARED_NOT_TIMING_PASS','COMPLETE_COMBINATIONAL_CONES_PREPARED_NOT_TIMING_PASS')
body=body.replace('canonical_sparse_snapshot_v3.py','canonical_sparse_snapshot_v4.py')
body=once(body,"r['rationale']=r['rationale'][:4]+plan['rationale']", "r['rationale']=[r['rationale'][0]]+plan['rationale']")
body=once(body,"Scope below0.5percent does not guarantee timing.","The reviewed5.74percent is a larger constraint set, not a timing or routability guarantee.")
body=once(body,"    r['reviewed_evidence'].update(plan['source_files'])", "    r['reviewed_evidence'].update(plan['source_files'])\n    r['reviewed_evidence'][str(HERE/'camera_sparse_route_v3.py')]=sha(HERE/'camera_sparse_route_v3.py')")
body=once(body,"enable_route=dict(net='local_driver_net_261',loads=18,identical_clock_reset_and_polarities=True,locations_unchanged=True)","enable_routes=plan['enable_review']")
prefix,guard,tail=body.rpartition("if __name__=='__main__':")
assert guard
exec(compile(prefix,__file__+':complete-input-cones','exec'),globals())
def fanouts():return build()['rows']
NETS=[r['net'] for r in fanouts()]
if __name__=='__main__':
    action=sys.argv.pop(1)
    if action=='prepare':prepare()
    elif action=='route':route()
    else:raise ValueError(action)
