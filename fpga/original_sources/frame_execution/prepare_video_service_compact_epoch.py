"""Full native compact candidate with stable nested legacy-page epoch."""
from pathlib import Path
source=Path(__file__).with_name('prepare_video_service_compact.py');code=source.read_text()
old='from video_service_compact_native import extension as serializer_extension'
assert code.count(old)==1
code=code.replace(old,'from video_service_compact_epoch_native import extension as serializer_extension')
old="HERE/'video_service_compact_native.py',"
assert code.count(old)==1
code=code.replace(old,"HERE/'video_service_compact_native.py',HERE/'video_service_compact_epoch_native.py',HERE/'prepare_video_service_compact.py',")
exec(compile(code,str(source)+':stable-camera-epoch','exec'),globals())
