"""Independent native12 candidate with flat diagnostic selection only."""
from pathlib import Path
source=Path(__file__).with_name('prepare_video_service_native.py');code=source.read_text()
old='from video_service_display_native import extension as display_extension,explicit_net_assignments,split_net_declarations'
new='from video_service_display_native import explicit_net_assignments,split_net_declarations\nfrom video_service_page_decode_native import extension as display_extension'
assert code.count(old)==1;code=code.replace(old,new)
old="[SOURCE,Path(__file__),HERE/'video_service_display_native.py'"
assert code.count(old)==1;code=code.replace(old,"[SOURCE,Path(__file__),HERE/'prepare_video_service_native.py',HERE/'video_service_page_decode_native.py',HERE/'video_service_display_native.py'")
exec(compile(code,str(source)+':flat-page-decode','exec'),globals())
