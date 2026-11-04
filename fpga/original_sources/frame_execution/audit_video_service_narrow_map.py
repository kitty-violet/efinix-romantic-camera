"""Same exact retained-primitive proof, with the explicit narrow epoch source."""
from pathlib import Path
source=Path(__file__).with_name('audit_video_service_epoch_map.py');outer=source.read_text()
outer=outer.replace("=='service_page_sequence[4]'","=='service_legacy_epoch[1]'")
assert outer!=source.read_text()
exec(compile(outer,str(source)+':narrow-epoch','exec'),globals())
