"""Source-bound service physical flow with optional CNN RAM locations released."""
from pathlib import Path
origin=Path(__file__).with_name('place_video_service_serial.py');outer=origin.read_text()
anchor="exec(compile(code,str(source)+':indexed-serializer','exec'),globals())"
change='''
needle=" code=once(code,'from local_control_decode import audit_delta','from place_video_service_serial import audit_delta')"
assert code.count(needle)==1
code=code.replace(needle,needle+"\\n code=code.replace('from fixed_cnn_ram_locations import','from floating_cnn_ram_locations import').replace('same DSP cascade and CNN RAM anchors, no general logic fixed.','same DSP cascade anchors; CNN RAM and ordinary logic free for global placement.')")
code=code.replace('hard_anchors=401','hard_anchors=288')
'''
assert outer.count(anchor)==1
outer=outer.replace(anchor,change+'\n'+anchor)
exec(compile(outer,str(origin)+':free-ram','exec'),globals())
