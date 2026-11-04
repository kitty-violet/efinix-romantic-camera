"""Preserve the reviewed measured cones as a group, leaving the failed cone free."""
import json,sys
from pathlib import Path
from run_units import HERE,sha
from control_command_repair import once
import camera_sparse_route_v2 as previous
from camera_route_cone_plan import build

body=previous.body.replace('camera-checked-bound7','camera-checked-bound8').replace('camera-checked-route-review7','camera-checked-route-review8')
body=body.replace('SEVEN_DATA_ONE_ENABLE_PRESERVATION_PREPARED_NOT_TIMING_PASS','MEASURED_DATA_CONES_AND_ONE_ENABLE_PREPARED_NOT_TIMING_PASS')
body=body.replace("HERE/'canonical_sparse_snapshot_v2.py'","HERE/'canonical_sparse_snapshot_v3.py'")
body=once(body,"    docs=EF/'doc/topics'",'''    plan=build()
    r['rationale']=r['rationale'][:4]+plan['rationale']
    r['remaining_risks']=['Restored measured cones may constrain other routes. Scope below0.5percent does not guarantee timing. Dedicated/control nets not selected stay free; all original whole-design gates remain mandatory.']
    r['reviewed_evidence'].update(plan['source_files'])
    r['reviewed_evidence'][str(HERE/'camera_sparse_route_v2.py')]=sha(HERE/'camera_sparse_route_v2.py')
    docs=EF/'doc/topics' ''')
body=once(body,"        rows=rows,excluded_failed_path_nets", "        cone_review=plan,rows=rows,excluded_failed_path_nets")
prefix,guard,tail=body.rpartition("if __name__=='__main__':")
assert guard
exec(compile(prefix,__file__+':reviewed-cone-adapter','exec'),globals())
def fanouts():return build()['rows']
NETS=[r['net'] for r in fanouts()]

if __name__=='__main__':
    action=sys.argv.pop(1)
    if action=='prepare':prepare()
    elif action=='route':route()
    else:raise ValueError(action)
