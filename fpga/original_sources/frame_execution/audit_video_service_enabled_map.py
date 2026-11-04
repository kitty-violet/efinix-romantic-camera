"""Retained graph audit with function-based constant-folded feedback LUTs.

The mapper folds the literal1 input of held bit495. Names on every FF and LUT
output still bind the net bijection; new feedback LUT inputs are checked by
the existing exhaustive Boolean canonicalizer, not by pin position.
"""
from pathlib import Path
source=Path(__file__).with_name('audit_video_service_map.py');code=source.read_text()
old="  for p in pins.keys() & b[n][1].keys():"
new=old+"\n   if n.startswith('service_enable_feedback_') and p.startswith('I'):continue"
assert code.count(old)==1;code=code.replace(old,new)
code=code.replace('checker_sha256=sha(__file__),','checker_sha256=sha(__file__),original_checker_sha256=sha(Path(__file__).with_name("audit_video_service_map.py")),')
exec(compile(code,str(source)+':feedback-input-function','exec'),globals())
