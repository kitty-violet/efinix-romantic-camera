"""Unchanged original Gray checks plus the exact schema5 mailbox geometry."""
from pathlib import Path
source=Path(__file__).with_name('video_service_gray.py')
code=source.read_text()
for old,new in [
 ('audit_video_service_compact_routed_cdc','audit_video_service_narrow_split_cdc'),
 ('audit_video_service_compact_jtag','audit_video_service_narrow_jtag'),
]:
 assert old in code,old
 code=code.replace(old,new)
exec(compile(code,str(source)+':schema5','exec'),globals())
