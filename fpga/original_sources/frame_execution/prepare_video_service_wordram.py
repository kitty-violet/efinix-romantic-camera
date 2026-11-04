"""RAM-backed native candidate with word-sized service observation."""
from pathlib import Path
original=Path(__file__).with_name('prepare_video_service_fixedram.py');outer=original.read_text()
anchor='code=source.read_text()'
extra='''
code=code.replace('from video_service_display_native import extension as display_extension,explicit_net_assignments,split_net_declarations','from video_service_word_native import extension as display_extension,explicit_net_assignments,split_net_declarations')
code=code.replace("(out/filename).write_bytes((VARIANT/filename).read_bytes())","(out/filename).write_bytes((HERE/'variants/video_service_word'/filename if filename=='video_debug_mailbox.sv' else VARIANT/filename).read_bytes())")
'''
assert outer.count(anchor)==1;outer=outer.replace(anchor,anchor+'\n'+extra)
outer=outer.replace('variants/video_service_ram_fixed/video_snapshot_serializer.sv','variants/video_service_ram_parallel/video_snapshot_serializer.sv')
outer=outer.replace("HERE/'video_service_fixedram_native.py',","HERE/'prepare_video_service_fixedram.py',HERE/'video_service_word_native.py',HERE/'variants/video_service_word/video_debug_mailbox.sv',HERE/'video_service_fixedram_native.py',")
outer=outer.replace('serializer_code,serializer_receipt=serializer_extension(D,removed,edits)','serializer_code,serializer_receipt=serializer_extension(D,removed,edits)\\n serializer_receipt.update(store_words=4,physical_ram_blocks=16,ram_writes_per_capture=64)')
exec(compile(outer,str(original)+':word-parallelram','exec'),globals())
