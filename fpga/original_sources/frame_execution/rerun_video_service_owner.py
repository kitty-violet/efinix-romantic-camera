"""Replay exact prepared owner fixtures after a license-only start failure."""
from pathlib import Path
source=Path(__file__).with_name('rerun_video_service_serializer.py');code=source.read_text()
changes=[
 ('work.tb_video_snapshot_serializer','work.tb_video_pair_owner'),
 ('PASS VIDEO_SERVICE_INDEXED_SERIAL_NATIVE','PASS ACTUAL_MAP_OWNER_BOUNDARY'),
 ('VIDEO_SERVICE_INDEXED_SERIAL_NATIVE_PASS','ACTUAL_MAP_OWNER_FAILURE_BOUNDARY_PASS'),
 (" (root/'inputs.json').write_text(json.dumps(manifest,indent=2))", """ manifest.update(map_sha256=meta['map_sha256'],selected_cells=meta['selected_cells'],replay_source_checker_sha256=sha(source))
 (root/'inputs.json').write_text(json.dumps(manifest,indent=2))"""),
 ("result.update(status='ACTUAL_MAP_OWNER_FAILURE_BOUNDARY_PASS',markers=", "result.update(status='ACTUAL_MAP_OWNER_FAILURE_BOUNDARY_PASS',selected_cells=len(meta['selected_cells']),markers="),
]
for old,new in changes:
 assert old in code,old
 code=code.replace(old,new)
exec(compile(code,str(source)+':exact-owner-fixture-replay','exec'),globals())
