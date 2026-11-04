"""Congestion-aware placement of the same map; always stop before routing."""
from pathlib import Path

parent = Path(__file__).with_name('continuous_control_place.py')
body = parent.read_text(encoding='utf8')
assert body.count('continuous-control-place1') == 1
body = body.replace('continuous-control-place1', 'continuous-control-c3-place1')
needle = "exec(compile(code,str(SOURCE)+':continuous-control-place','exec'),globals())"
assert body.count(needle) == 1
body = body.replace(needle, "assert code.count(\"'CONGESTION_2'\")==1\ncode=code.replace(\"'CONGESTION_2'\",\"'CONGESTION_3'\")\n" + needle)
exec(compile(body, str(parent) + ':c3-placement', 'exec'), globals())
