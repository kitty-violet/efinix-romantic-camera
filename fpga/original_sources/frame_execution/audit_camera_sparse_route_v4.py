"""Strict source, topology, shared-control and geometry checks for complete cones."""
from run_units import HERE
from control_command_repair import once
import audit_camera_sparse_route_v3 as previous
body=previous.body.replace('camera_sparse_route_v3','camera_sparse_route_v4')
body=body.replace('MEASURED_DATA_CONES_AND_ONE_ENABLE_PREPARED_NOT_TIMING_PASS','COMPLETE_COMBINATIONAL_CONES_PREPARED_NOT_TIMING_PASS')
body=body.replace('200<len(NETS)<400','2000<len(NETS)<5000')
body=body.replace('from camera_route_cone_plan import build','from camera_route_cone_plan_v2 import build')
body=once(body,"assert rule['enable_route']==dict(net='local_driver_net_261',loads=18,identical_clock_reset_and_polarities=True,locations_unchanged=True)","assert rule['enable_routes']==expected_plan['enable_review']")
prefix,guard,tail=body.rpartition("if __name__=='__main__':")
assert guard
exec(compile(prefix,__file__+':strict-complete-cone-audit','exec'),globals())
if __name__=='__main__':
    import textwrap
    exec(compile(textwrap.dedent(tail),__file__+':cli','exec'),globals())
