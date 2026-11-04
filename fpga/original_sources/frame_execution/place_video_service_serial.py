"""Run the unchanged physical flow with a bound current native13 audit."""
from pathlib import Path
source=Path(__file__).with_name('place_video_service.py')
code=source.read_text()
old=" assert graph['status']=='VIDEO_SERVICE_RETAINED_MAPPED_GRAPH_PASS' and graph['retained']==58312"
new=""" native=Path(meta['native']);native_proof=json.loads((native/'native-graph-audit.json').read_text())
 assert sha(native/'native-graph-audit.json')==meta['native_audit_sha256']
 assert native_proof['status']=='SERVICE_NATIVE_RETAINED_GRAPH_AND_CUT_PASS'
 for f,h in native_proof['inputs'].items():assert sha(f)==h,f
 assert graph['status']=='VIDEO_SERVICE_RETAINED_MAPPED_GRAPH_PASS'
 assert graph['retained']==native_proof['retained_primitives']"""
assert code.count(old)==1
code=code.replace(old,new).replace('from place_video_service import audit_delta','from place_video_service_serial import audit_delta')
exec(compile(code,str(source)+':indexed-serializer','exec'),globals())
