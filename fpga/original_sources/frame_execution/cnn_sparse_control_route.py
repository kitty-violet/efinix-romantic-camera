"""Original C2 routing and complete isolated original-constraint STA gates."""
from pathlib import Path
from control_command_repair import once
source=Path(__file__).with_name('cnn_observer_trim_free_route.py')
body=source.read_text(encoding='utf8')
body=body.replace('cnn_observer_trim_board','cnn_sparse_control_board')
body=body.replace('cnn_observer_trim_implementation','cnn_sparse_control_implementation')
body=once(body,"    compile(code,__file__,'exec');return source,code", """    if fn==original.route_code:
        code=once(code,"HERE/'canonical_snapshot_sta.py'","HERE/'canonical_snapshot_sta_isolated.py'")
        code=once(code,"'--root',canonical],ROOT,300)","'--root',canonical],ROOT,450)")
    compile(code,__file__,'exec');return source,code""")
exec(compile(body,str(source)+':sparse-control-route','exec'),globals())
