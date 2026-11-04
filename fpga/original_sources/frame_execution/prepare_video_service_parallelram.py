"""Same fixed-source atomic snapshot,16 RAM lanes/four writes per bank."""
from pathlib import Path
original=Path(__file__).with_name('prepare_video_service_fixedram.py')
code=original.read_text().replace('variants/video_service_ram_fixed/video_snapshot_serializer.sv','variants/video_service_ram_parallel/video_snapshot_serializer.sv')
code=code.replace("HERE/'video_service_fixedram_native.py',", "HERE/'prepare_video_service_fixedram.py',HERE/'video_service_fixedram_native.py',")
code=code.replace('serializer_code,serializer_receipt=serializer_extension(D,removed,edits)',
 'serializer_code,serializer_receipt=serializer_extension(D,removed,edits)\\n serializer_receipt.update(store_words=4,physical_ram_blocks=16,ram_writes_per_capture=64)')
exec(compile(code,str(original)+':parallel16','exec'),globals())
