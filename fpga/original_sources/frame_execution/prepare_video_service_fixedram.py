"""Independent native15: constant-position source capture and banked RAM scan."""
from pathlib import Path
source=Path(__file__).with_name('prepare_video_service_native.py')
code=source.read_text()
changes=[
 ('from video_service_modes_native import extension as modes_extension',
  'from video_service_modes_native import extension as modes_extension\nfrom video_service_fixedram_native import extension as serializer_extension'),
 ("[SOURCE,Path(__file__),HERE/'video_service_display_native.py'",
  "[SOURCE,Path(__file__),HERE/'prepare_video_service_native.py',HERE/'video_service_serializer_native.py',HERE/'video_service_fixedram_native.py',HERE/'variants/video_service_ram_fixed/video_snapshot_serializer.sv',HERE/'video_service_display_native.py'"),
 (' modes_code,modes_receipt=modes_extension(D,removed,edits)',
  ' modes_code,modes_receipt=modes_extension(D,removed,edits)\n serializer_code,serializer_receipt=serializer_extension(D,removed,edits,supplied)'),
 (" code+='\\n'+display_code+'\\n'+modes_code",
  " old_request='assign service_snapshot_request='+v('cpu/system_hardJtag_debug_logic_jtagBridge/observation/n5')+';'\n assert display_code.count(old_request)==1,old_request\n display_code=display_code.replace(old_request,'assign service_snapshot_request=service_snapshot_capture;')\n code+='\\n'+display_code+'\\n'+modes_code+'\\n'+serializer_code"),
 (" report=dict(status=", " (out/'video_snapshot_serializer.sv').write_bytes((HERE/'variants/video_service_ram_fixed/video_snapshot_serializer.sv').read_bytes())\n report=dict(status="),
 ('display_extension=display_receipt,modes_extension=modes_receipt)',
  'display_extension=display_receipt,modes_extension=modes_receipt,serializer_extension=serializer_receipt)')]
for old,new in changes:
 assert code.count(old)==1,old
 code=code.replace(old,new)
exec(compile(code,str(source)+':fixedram','exec'),globals())
