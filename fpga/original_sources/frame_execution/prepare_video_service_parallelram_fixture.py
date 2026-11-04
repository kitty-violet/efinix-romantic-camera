"""Existing entire actual-observer differential test, parallel RAM variant."""
from pathlib import Path
original=Path(__file__).with_name('prepare_video_service_fixedram_fixture.py')
code=original.read_text().replace('variants/video_service_ram_fixed/video_snapshot_serializer.sv','variants/video_service_ram_parallel/video_snapshot_serializer.sv')
exec(compile(code,str(original)+':parallel16','exec'),globals())
