"""Eight frozen routes, including the separately reviewed synchronous CE tree."""
from pathlib import Path
from run_units import HERE,sha
from control_command_repair import once

body=(HERE/'audit_camera_sparse_route.py').read_text()
body=body.replace('from camera_sparse_route import','from camera_sparse_route_v2 import')
body=body.replace("HERE/'camera_sparse_route.py'","HERE/'camera_sparse_route_v2.py'")
body=body.replace('SEVEN_ORDINARY_NET_PRESERVATION_PREPARED_NOT_TIMING_PASS','SEVEN_DATA_ONE_ENABLE_PRESERVATION_PREPARED_NOT_TIMING_PASS')
body=body.replace('len(NETS)==7','len(NETS)==8').replace('7/79318','8/79318').replace('nets=7','nets=8')
body=once(body,"    assert rule['rows']==json.loads(json.dumps(fanouts()))",
    "    assert rule['enable_route']==dict(net='local_driver_net_261',loads=18,identical_clock_reset_and_polarities=True,locations_unchanged=True)\n    assert rule['rows']==json.loads(json.dumps(fanouts()))")
prefix,guard,tail=body.rpartition("if __name__=='__main__':")
assert guard
exec(compile(prefix,__file__+':v1-gates','exec'),globals())
