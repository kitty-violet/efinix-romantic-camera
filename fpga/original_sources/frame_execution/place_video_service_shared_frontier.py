"""Original physical gates with the exact current proven mapped-pruning count."""
from pathlib import Path
source=Path(__file__).with_name('place_video_service_epoch.py');code=source.read_text()
old="assert graph['retained']+1==native_proof['retained_primitives']"
new="assert graph['collapsed_identity_luts'][0]['all_aliases_four_state_proved']\\n assert graph['retained']+1+len(graph['collapsed_identity_luts'][0]['cells'])==native_proof['retained_primitives']"
assert code.count(old)==1;code=code.replace(old,new).replace('from place_video_service_epoch import audit_delta','from place_video_service_shared_frontier import audit_delta')
exec(compile(code,str(source)+':shared-frontier','exec'),globals())
