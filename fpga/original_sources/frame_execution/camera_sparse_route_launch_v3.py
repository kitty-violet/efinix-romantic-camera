"""Run one bounded whole-design route after the combined-cone review."""
from run_units import HERE
from control_command_repair import once
body=(HERE/'camera_sparse_route_launch.py').read_text()
body=body.replace('import camera_sparse_route_v2 as prepared','import camera_sparse_route_v3 as prepared')
body=body.replace('audit_camera_sparse_route_v2.py','audit_camera_sparse_route_v3.py').replace('canonical_sparse_snapshot_v2.py','canonical_sparse_snapshot_v3.py')
body=once(body,"for f in (Path(__file__),", "for f in (Path(__file__),HERE/'camera_route_cone_plan.py',HERE/'camera_sparse_route_v3.py',HERE/'camera_sparse_route_v2.py',HERE/'camera_sparse_route.py',")
exec(compile(body,__file__+':reviewed-launch','exec'),globals())
