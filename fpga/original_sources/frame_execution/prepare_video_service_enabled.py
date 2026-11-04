"""Independent candidate: native12 functional changes, observation CE folding."""
from pathlib import Path
source=Path(__file__).with_name('prepare_video_service_native.py');code=source.read_text()
changes=[
 ('from video_service_modes_native import extension as modes_extension','from video_service_modes_native import extension as modes_extension\nfrom video_service_enable_native import extension as enable_extension'),
 ("[SOURCE,Path(__file__),HERE/'video_service_display_native.py'","[SOURCE,Path(__file__),HERE/'prepare_video_service_native.py',HERE/'video_service_enable_native.py',HERE/'video_service_display_native.py'"),
 (' modes_code,modes_receipt=modes_extension(D,removed,edits)',' modes_code,modes_receipt=modes_extension(D,removed,edits)\n enable_code,enable_receipt=enable_extension(D,removed,edits)'),
 (" code+='\\n'+display_code+'\\n'+modes_code"," code+='\\n'+display_code+'\\n'+modes_code+'\\n'+enable_code"),
 ('display_extension=display_receipt,modes_extension=modes_receipt)','display_extension=display_receipt,modes_extension=modes_receipt,enable_extension=enable_receipt)')]
for old,new in changes:
 assert code.count(old)==1,old
 code=code.replace(old,new)
exec(compile(code,str(source)+':enable-fold','exec'),globals())
