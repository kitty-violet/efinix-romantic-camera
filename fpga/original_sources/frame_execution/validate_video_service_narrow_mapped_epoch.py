"""Actual nested selector cones, stable C/E/D choice across four32-bit words."""
from pathlib import Path
source=Path(__file__).with_name('validate_video_service_mapped_epoch.py');code=source.read_text()
changes=[
 ('from video_service_sample_ram import capture_roots','from video_service_narrow_sample_ram import capture_roots'),
 ('if i<128','if i<32'),('len(roots)==128','len(roots)==32'),
 ("oldvals[net(P+'request_sync')]=seq>>3&1;oldvals[net('cnn_run_seen')]=seq>>4&1", "epoch=(seq//2)&3;oldvals[net(P+'request_sync')]=epoch&1;oldvals[net('cnn_run_seen')]=(epoch>>1)&1"),
 ('for chunk in (6,7):','for chunk in (26,27,28,29):'),
 ("('service_page_part',2,3),('service_page_sequence',8,seq+chunk-6)","('service_page_part',4,chunk//2),('service_page_sequence',8,(seq+chunk)&255),('service_legacy_epoch',2,epoch)"),
 ('start=64 if chunk==6 else 0','start=0'),
 ('range(start,start+64))<<(64*(chunk-6))','range(32))<<(32*(chunk-26))')]
for old,new in changes:assert old in code,old;code=code.replace(old,new)
exec(compile(code,str(source)+':narrow-epoch-oracle','exec'),globals())
