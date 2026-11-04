"""One fresh full placement using the actually passed macro floorplan."""
from pathlib import Path
source=Path(__file__).with_name('place_video_sensor73.py');code=source.read_text()
code=code.replace("from video_sensor73_physical import apply_profile","from video_sensor73_macro_physical import apply_profile")
code=code.replace("from video_sensor73_physical import generate as retain_cnn_ram","from video_sensor73_macro_physical import generate as retain_cnn_ram")
code=code.replace("Path(__file__).with_name('video_sensor73_physical.py')","Path(__file__).with_name('video_sensor73_macro_physical.py')")
code=code.replace('hard_anchors=404','hard_anchors=533')
assert code!=source.read_text()
exec(compile(code,str(source)+':whole-macro-floorplan','exec'),globals())
