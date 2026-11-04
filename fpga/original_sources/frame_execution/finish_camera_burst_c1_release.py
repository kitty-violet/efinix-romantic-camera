"""Same complete release gates, with truthful C1 route-wrapper provenance."""
from pathlib import Path
from control_command_repair import once

source=Path(__file__).with_name('finish_camera_burst_release.py')
code=source.read_text(encoding='utf8')
code=code.replace('finish_camera_burst_release.py','finish_camera_burst_c1_release.py')
code=once(code,".replace('finish_camera_recovery_release','finish_camera_burst_release')", ".replace('finish_camera_recovery_release','finish_camera_burst_c1_release')")
code=once(code," return once(prepare_code()[1],'from camera_checked_board import audit_delta','from camera_capture_burst_board import audit_delta')", " from camera_burst_c1_route import prepare_code as c1_prepare\n return c1_prepare()[1]")
anchor="exec(compile(code,__file__+':full-hardware-gates','exec'),globals())"
replacement="""code=once(code,\"wrappers(batch,[('-local-driver-route',HERE/'camera_burst_flow.py')],bind,verify)\",\"wrappers(batch,[('-local-driver-route',HERE/'camera_burst_c1_route.py')],bind,verify)\")
code=once(code,\"str(HERE/'camera_burst_flow.py'):pm['adapter_sha256']\",\"str(HERE/'camera_burst_c1_route.py'):pm['adapter_sha256']\")
"""+anchor
code=once(code,anchor,replacement)
exec(compile(code,str(source)+':c1-provenance','exec'),globals())
