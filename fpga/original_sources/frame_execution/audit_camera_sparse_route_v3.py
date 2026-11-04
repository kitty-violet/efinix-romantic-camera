"""Validate measured-cone selection, immutable traces, geometry and tool uptake."""
from run_units import HERE
from control_command_repair import once
body=(HERE/'audit_camera_sparse_route.py').read_text()
body=body.replace('from camera_sparse_route import','from camera_sparse_route_v3 import')
body=body.replace("HERE/'camera_sparse_route.py'","HERE/'camera_sparse_route_v3.py'")
body=body.replace('SEVEN_ORDINARY_NET_PRESERVATION_PREPARED_NOT_TIMING_PASS','MEASURED_DATA_CONES_AND_ONE_ENABLE_PREPARED_NOT_TIMING_PASS')
body=body.replace('len(NETS)==7','200<len(NETS)<400').replace("'7/79318 nets will be constrained (reloaded)'","str(len(NETS))+'/79318 nets will be constrained (reloaded)'").replace('nets=7','nets=len(NETS)')
body=once(body,"    assert rule['rows']==json.loads(json.dumps(fanouts()))",'''    from camera_route_cone_plan import build
    expected_plan=json.loads(json.dumps(build()))
    assert rule['cone_review']==expected_plan
    verify(expected_plan['source_files'])
    assert rule['enable_route']==dict(net='local_driver_net_261',loads=18,identical_clock_reset_and_polarities=True,locations_unchanged=True)
    assert rule['rows']==json.loads(json.dumps(fanouts()))''')
prefix,guard,tail=body.rpartition("if __name__=='__main__':")
assert guard
exec(compile(prefix,__file__+':v1-gates','exec'),globals())
if __name__=='__main__':
    import textwrap
    exec(compile(textwrap.dedent(tail),__file__+':cli','exec'),globals())
