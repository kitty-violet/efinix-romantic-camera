"""Use the same whole-chip review with the explicit533-macro legality gate."""
from pathlib import Path
source=Path(__file__).with_name('review_video_sensor73.py');code=source.read_text()
old='from video_sensor73_physical import verify_placed,BOARD,REFERENCE'
assert code.count(old)==1
code=code.replace(old,'from video_sensor73_macro_physical import verify_placed,BOARD,REFERENCE')
exec(compile(code,str(source)+':whole-macro-floorplan','exec'),globals())
