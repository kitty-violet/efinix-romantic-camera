"""Keep joint decoder remap, restrict replicas to measured slow path nets.

The broad native6 placement increased whole-chip wire/congestion demand;
this independently generated alternative does not alter that frozen evidence.
"""
from pathlib import Path
BASE_SCRIPT=Path(__file__).with_name('camera_joint_native.py')
code=BASE_SCRIPT.read_text()
old="if not ((len(ls)>=128 and extent>=70) or (label(tok) in hot and (extent>=45 or distance>=70))):continue"
assert code.count(old)==1
code=code.replace(old,"if not (label(tok) in hot and (extent>=45 or distance>=70)):continue")
needle="for i in (1,2):\n  f=ROOT/f'tmp/frame-execution/camera-global-survey{i}b/outflow/canonical_setup.rpt'"
assert code.count(needle)==1
code=code.replace(needle,"r['inputs'][str(BASE_SCRIPT)]=sha(BASE_SCRIPT)\n "+needle.lstrip())
exec(compile(code,str(BASE_SCRIPT)+':measured-critical-only','exec'),globals())
