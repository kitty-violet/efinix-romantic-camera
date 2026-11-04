"""Same frozen ISP logic/seed/anchors; compare C1 global placement and packing.

C2 routed legally but failed 509 distinct setup endpoints, mostly retained CNN
RAM/control/data routes. This changes the global physical recipe, not a seed,
individual gate location, timing exception, clock or logical implementation.
Stop at placement and review the whole device before deciding whether to route.
"""
from pathlib import Path

source = Path(__file__).with_name('place_video_input_isp.py')
text = source.read_text()
needle = "exec(compile(code,str(source)+':input-isp-proof','exec'),globals())"
assert text.count(needle) == 1
insert = """old_recipe=\"'--optimization','CONGESTION_2','--seed','3'\"
assert code.count(old_recipe)==1
code=code.replace(old_recipe,\"'--optimization','CONGESTION_1','--seed','3'\")
"""
text = text.replace(needle, insert + needle)
exec(compile(text, str(source) + ':C1-global-comparison', 'exec'), globals())
