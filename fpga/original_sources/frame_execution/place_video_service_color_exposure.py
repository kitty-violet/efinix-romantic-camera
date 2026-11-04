"""Use the established bounded placement flow with the explicit color delta proof.

Does not relabel native67 as unchanged64. Stops after placement for congestion
review; old route and old STA never release this candidate.
"""
from pathlib import Path

source=Path(__file__).with_name('place_video_service.py')
code=source.read_text()
start=code.index('def audit_delta(board):')
end=code.index('\ndef main():',start)
audit='''def audit_delta(board):
 meta=json.loads((board/'inputs.json').read_text());result=json.loads((board/'result.json').read_text())
 assert result['status']=='VIDEO_SERVICE_COMPLETE_NATIVE_SYNTHESIS_PROBE_PASS'
 assert sha(board/'inputs.json')==result['inputs_sha256']
 for r in meta['inputs']:assert sha(r['snapshot'])==r['sha256'],r['snapshot']
 proof=json.loads((board/'color-map-audit.json').read_text())
 assert proof['status']=='COLOR_EXPOSURE_CURRENT_MAP_COMPOSITION_PASS'
 assert proof['all_other_public_cells_and_parameters_identical'] and proof['changed_public_cells']==17
 assert proof['source_sha256']==sha(board/'native_source.v')
 assert proof['map_sha256']==result['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v')
 assert proof['vdb_sha256']==result['vdb_sha256']==sha(board/'outflow/ti60f225_oob.vdb')
 for f,h in proof['evidence'].items():assert sha(f)==h,f
 arithmetic=ROOT/'tmp/frame-execution/video-service-color-arithmetic1'
 numeric=json.loads((arithmetic/'result.json').read_text())
 assert numeric['status']=='ACTUAL_MAPPED_BAYER_EXHAUSTIVE_ARITHMETIC_PASS' and numeric['comparisons']==131072
 assert numeric['inputs_sha256']==sha(arithmetic/'inputs.json')
 for f,h in json.loads((arithmetic/'inputs.json').read_text())['evidence'].items():assert sha(f)==h
 from video_service_physical_profile import periphery_audit
 periphery_audit(board)
 return dict(status='COLOR_EXPOSURE_MAP_BOUND_FOR_PLACEMENT_NOT_RELEASE',map_sha256=result['map_sha256'])
'''
code=code[:start]+audit+code[end:]
code=code.replace('from place_video_service import audit_delta','from place_video_service_color_exposure import audit_delta')
code=code.replace("board/'retained-map-audit.json'","board/'color-map-audit.json'")
exec(compile(code,str(source)+':color-exposure-proof','exec'),globals())
