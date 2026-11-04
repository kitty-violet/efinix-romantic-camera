"""Version5 actual-map transport with all control/CDC/scan cells retained."""
from pathlib import Path
source=Path(__file__).with_name('prepare_video_service_compact_mapped_serial.py');code=source.read_text()
for old,new in [
 ('from video_service_sample_ram import capture_roots','from video_service_narrow_sample_ram import capture_roots'),
 ('len(held) in (147,192)','len(held)==96'),
 ('[1:0]fixture_part','[3:0]fixture_part'),('[182:0]fixture_packet','[86:0]fixture_packet'),
 ('[1:0]service_page_part','[3:0]service_page_part'),('[182:0]service_observation_packet','[86:0]service_observation_packet'),
 ('[191:0]','[95:0]')]:
 assert old in code,old;code=code.replace(old,new)
exec(compile(code,str(source)+':narrow-map-transport','exec'),globals())
