"""Bounded official ISP placement. Stop before route for global review."""
from pathlib import Path

source=Path(__file__).with_name('place_video_service.py');code=source.read_text()
start=code.index('def audit_delta(board):');end=code.index('\ndef main():',start)
audit='''def audit_delta(board):
 meta=json.loads((board/'inputs.json').read_text());result=json.loads((board/'result.json').read_text())
 assert result['status']=='VIDEO_SERVICE_COMPLETE_NATIVE_SYNTHESIS_PROBE_PASS'
 assert sha(board/'inputs.json')==result['inputs_sha256']
 for r in meta['inputs']:assert sha(r['snapshot'])==r['sha256'],r['snapshot']
 proof=json.loads((board/'input-isp-map-audit.json').read_text())
 assert proof['status']=='INPUT_ISP_SOURCE_AND_RETAINED_MAP_PASS' and proof['retained']==54240
 assert proof['source_sha256']==sha(board/'native_source.v')
 assert proof['map_sha256']==result['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v')
 assert proof['vdb_sha256']==result['vdb_sha256']==sha(board/'outflow/ti60f225_oob.vdb')
 for f,h in proof['evidence'].items():assert sha(f)==h,f
 tests=['video-input-color4','video-input-lut2','video-input-mapped1','video-input-native2','video-input-continuous-isp2','video-input-recovery-wide3']
 evidence={}
 for t in tests:
  root=ROOT/'tmp/frame-execution'/t
  value=json.loads((root/'result.json').read_text());assert value['status']=='PASS',t
  assert value['inputs_sha256']==sha(root/'inputs.json'),t
  evidence[str(root/'result.json')]=sha(root/'result.json')
  evidence[str(root/'inputs.json')]=sha(root/'inputs.json')
 assert sha(board/'video_input_lut.sv')==sha(ROOT/'tmp/frame-execution/video-input-mapped1/video_input_lut.sv')==sha(ROOT/'tmp/frame-execution/video-input-continuous-isp2/video_input_lut.sv')
 from video_service_physical_profile import periphery_audit
 periphery_audit(board)
 return dict(status='INPUT_ISP_MAP_BOUND_FOR_PLACEMENT_NOT_RELEASE',map_sha256=result['map_sha256'],evidence=evidence)
'''
code=code[:start]+audit+code[end:]
code=code.replace('from place_video_service import audit_delta','from place_video_input_isp import audit_delta')
code=code.replace("board/'retained-map-audit.json'","board/'input-isp-map-audit.json'")
# Carry the initialized RAM data into the implementation snapshot even though
# placement consumes the official VDB and does not resynthesize.
old="source.read_text().replace('from windows_process_tree import stop_launched_tree','from windows_process_tree_safe import stop_launched_tree')"
new=old+".replace(\"('.xml','.sdc','.bin','.v','.sv','.vh','.svh')\",\"('.xml','.sdc','.bin','.v','.sv','.vh','.svh','.hex')\")"
assert code.count(old)==1
# This text sits inside a quoted generated Python source literal.
code=code.replace(old,new.replace('"','\\"'))
exec(compile(code,str(source)+':input-isp-proof','exec'),globals())
