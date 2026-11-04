"""Keep compact RAM snapshots, flatten the redundant service-page mux tree.

This changes only the combinational page selection. All fields, selector
encodings, clocks, captured edges and cross-domain handshakes are unchanged.
The original flat decoder proves all32 phase/part combinations, including
unused phase values. Current actual-map proofs must still be rerun.
"""
from pathlib import Path

source=Path(__file__).with_name('prepare_video_service_compact_epoch.py')
outer=source.read_text()
anchor="exec(compile(code,str(source)+':stable-camera-epoch','exec'),globals())"
injection=r'''
inner_anchor="exec(compile(code,str(source)+':compact-paged-transport','exec'),globals())"
adapt="""
old='from video_service_display_native import extension as display_extension,explicit_net_assignments,split_net_declarations'
new='from video_service_display_native import explicit_net_assignments,split_net_declarations\\nfrom video_service_page_decode_native import extension as display_extension'
assert code.count(old)==1;code=code.replace(old,new)
old="HERE/'video_service_native_binding.py',"
assert code.count(old)==1
code=code.replace(old,"HERE/'video_service_page_decode_native.py',HERE/'prepare_video_service_compact_epoch.py',HERE/'video_service_native_binding.py',")
"""
assert code.count(inner_anchor)==1
code=code.replace(inner_anchor,adapt+'\n'+inner_anchor)
'''
assert outer.count(anchor)==1
outer=outer.replace(anchor,injection+'\n'+anchor)
exec(compile(outer,str(source)+':flat-service-page','exec'),globals())
