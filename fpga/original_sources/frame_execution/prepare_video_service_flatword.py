"""Separate schema5 candidate; never edits the running narrow implementation."""
from pathlib import Path
import video_service_narrow_native as narrow
import video_service_flatword_native as flat
narrow.display=flat.display
narrow.serializer=flat.serializer
source=Path(__file__).with_name('prepare_video_service_narrow.py')
code=source.read_text()
old="HERE/'video_service_narrow_native.py',HERE/'video_service_compact_epoch_native.py'"
assert code.count(old)==1
code=code.replace(old,"HERE/'video_service_flatword_native.py',HERE/'prepare_video_service_narrow.py',HERE/'video_service_narrow_native.py',HERE/'video_service_compact_epoch_native.py'")
exec(compile(code,str(source)+':flat-combined-word','exec'),globals())
