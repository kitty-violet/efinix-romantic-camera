"""Recheck the inherited data-only reset change on this exact mapped build."""
from pathlib import Path
source=Path(__file__).with_name('validate_video_service_data_reset_mapped.py');code=source.read_text()
old=" receipt=n/'data-reset-edit.json';r=json.loads(receipt.read_text())"
new=""" local=json.loads((n/'weight-lookahead-edit.json').read_text())
 parent=Path(local['parent']);assert parent.name=='video-service-native42-data-reset-only'
 assert sha(parent/'native_source.v')==local['source_sha256']
 receipt=parent/'data-reset-edit.json';r=json.loads(receipt.read_text())"""
assert code.count(old)==1;code=code.replace(old,new)
old="[Path(__file__),receipt,b/'retained-map-audit.json'"
new="[Path(__file__),source,n/'weight-lookahead-edit.json',receipt,b/'retained-map-audit.json'"
assert code.count(old)==1;code=code.replace(old,new)
exec(compile(code,str(source)+':weight-current-map','exec'),globals())
