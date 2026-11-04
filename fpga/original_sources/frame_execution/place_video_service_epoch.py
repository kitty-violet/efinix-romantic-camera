"""Same bounded placement with exact mapped-pruning proof bound."""
from pathlib import Path
source=Path(__file__).with_name('place_video_service_serial.py');code=source.read_text()
old=" assert graph['retained']==native_proof['retained_primitives']"
new=" assert graph['discarded_unloaded_diagnostic_cells']==['cnn_run_seen~FF']\n assert graph['retained']+1==native_proof['retained_primitives']"
assert code.count(old)==1;code=code.replace(old,new)
code=code.replace('from place_video_service_serial import audit_delta','from place_video_service_epoch import audit_delta')
exec(compile(code,str(source)+':stable-epoch','exec'),globals())
