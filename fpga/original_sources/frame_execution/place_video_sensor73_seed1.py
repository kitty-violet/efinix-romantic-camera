"""One fresh seed1 placement with the original404 anchors and unchanged73 logic.

Keep all original constraints and stop before routing for the whole-chip review.
The previous seed3 legal routes lacked setup margin; its best seed does not
transfer automatically after the mapped design changes (Efinity timing guide).
"""
from pathlib import Path
source=Path(__file__).with_name('place_video_sensor73.py');code=source.read_text()
for old,new in [('seed=3,optimization=', 'seed=1,optimization='),("'--seed','3'", "'--seed','1'")]:
 assert code.count(old)==1,old
 code=code.replace(old,new)
exec(compile(code,str(source)+':seed1-original-404-anchor-plan','exec'),globals())
