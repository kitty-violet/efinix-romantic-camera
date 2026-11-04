"""Generate a separate 32-bit observation candidate, keeping the current core."""
from pathlib import Path
source=Path(__file__).with_name('prepare_video_service_compact.py');outer=source.read_text()
anchor="exec(compile(code,str(source)+':compact-paged-transport','exec'),globals())"
inject=r'''
old='from video_service_compact_native import extension as serializer_extension'
assert code.count(old)==1
code=code.replace(old,'from video_service_narrow_native import serializer as serializer_extension')
old='from video_service_display_native import extension as display_extension,explicit_net_assignments,split_net_declarations'
assert code.count(old)==1
code=code.replace(old,'from video_service_display_native import explicit_net_assignments,split_net_declarations\nfrom video_service_narrow_native import display as display_extension')
old="HERE/'video_service_native_binding.py',"
assert code.count(old)==1
code=code.replace(old,"HERE/'video_service_narrow_native.py',HERE/'video_service_compact_epoch_native.py',HERE/'prepare_video_service_compact.py',HERE/'video_service_native_binding.py',")
old=' report=dict(status='
patch="""
 narrow=out/'video_snapshot_serializer.sv'
 content=narrow.read_text()
 assert content.count('[191:0]')==2 and content.count('.WIDTH(192)')==1
 content=content.replace('[191:0]','[95:0]').replace('.WIDTH(192)','.WIDTH(96)')
 content=content.replace('shifter<={published,header};',"shifter<={96'd0,published,header};")
 narrow.write_text(content)
 mailbox=out/'video_debug_mailbox.sv';content=mailbox.read_text()
 assert content.count('[1:0]')==3
 content=content.replace('[1:0]','[PART_BITS-1:0]')
 content=content.replace('SOURCE_WIDTH=WIDTH)(', 'SOURCE_WIDTH=WIDTH,PART_BITS=(SOURCE_WIDTH/WIDTH<=1 ? 1:$clog2(SOURCE_WIDTH/WIDTH)))(')
 content=content.replace('SOURCE_WIDTH!=WIDTH && SOURCE_WIDTH!=4*WIDTH','SOURCE_WIDTH%WIDTH || (1<<PART_BITS)<SOURCE_WIDTH/WIDTH')
 mailbox.write_text(content)
"""
assert code.count(old)==1;code=code.replace(old,patch+'\n'+old)
'''
assert outer.count(anchor)==1;outer=outer.replace(anchor,inject+'\n'+anchor)
exec(compile(outer,str(source)+':narrow-observation','exec'),globals())
