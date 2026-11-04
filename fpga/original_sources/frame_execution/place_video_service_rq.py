"""Original placement gates with the independently proved RQ identity count."""
from pathlib import Path
source = Path(__file__).with_name('place_video_service_epoch.py')
code = source.read_text()
old = "assert graph['retained']+1==native_proof['retained_primitives']"
new = "assert [r['cell'] for r in graph['collapsed_identity_luts']]==['LUT__140273']\\n assert graph['collapsed_identity_luts'][0]['upstream_never_z']\\n assert graph['retained']+2==native_proof['retained_primitives']"
assert code.count(old) == 1
code = code.replace(old, new).replace('from place_video_service_epoch import audit_delta', 'from place_video_service_rq import audit_delta')
exec(compile(code, str(source)+':rq-identity', 'exec'), globals())
