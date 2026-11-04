"""Complete the same84 refinement with a measured, sufficient host budget.

The first attempt was stopped at87 annealing iterations by its945s host timer,
not by a functional/placement failure. It did not write a new place checkpoint.
Keep its evidence immutable. This copy changes only destination and wall budget.
"""
from pathlib import Path
source=Path(__file__).with_name('refine_video_control84_margin.py')
code=source.read_text(encoding='utf8')
for old,new in [
 ("DEST=B/'video-control84-refine3-c2margin'","DEST=B/'video-control84-refine4-c2margin'"),
 ('tool_budget_s=900,host_budget_s=945','tool_budget_s=1800,host_budget_s=1845'),
 ('--timeout 900','--timeout 1800'),
 ('budget_s=945','budget_s=1845'),
 ('proc.wait(timeout=945)','proc.wait(timeout=1845)'),
 ('refs=[Path(__file__).resolve(),PROOF,',"refs=[Path(__file__).resolve(),source.resolve(),B/'video-control84-refine3-c2margin/failure.json',B/'video-control84-refine3-c2margin/timeout-stop.json',PROOF,"),
 ]:
 assert code.count(old)==1,(old,code.count(old));code=code.replace(old,new)
exec(compile(code,str(source)+':measured1800s-budget','exec'),globals())
