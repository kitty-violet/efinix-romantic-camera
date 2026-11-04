"""Same frozen69 map/seed/C2; keep new ISP RAMs in the video-input region.

Do not resume or alter failed C1. Stop at placement for whole-device review.
"""
from pathlib import Path

source=Path(__file__).with_name('place_video_input_isp.py')
text=source.read_text()
needle="exec(compile(code,str(source)+':input-isp-proof','exec'),globals())"
assert text.count(needle)==1
insert='''anchor=" code=once(code,'from audit_global_place import metrics','from camera_ratio_before_place import metrics')"
assert code.count(anchor)==1
code=code.replace(anchor,anchor+"\\n code=code.replace('from fixed_cnn_ram_locations','from input_isp_local_ram_locations')")
assert code.count('hard_anchors=401')==1
code=code.replace('hard_anchors=401','hard_anchors=404')
'''
text=text.replace(needle,insert+needle)
exec(compile(text,str(source)+':input-local-RAM','exec'),globals())
