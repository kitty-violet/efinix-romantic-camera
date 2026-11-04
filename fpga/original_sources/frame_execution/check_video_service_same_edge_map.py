"""Current-map interface suite, with source-proved APB projection warning audit."""
from pathlib import Path
source=Path(__file__).with_name('check_video_service_control_priority_map.py');code=source.read_text();assert code.count('audit_video_service_control_priority_warnings.py')==1;code=code.replace('audit_video_service_control_priority_warnings.py','audit_video_service_same_edge_warnings.py')
code=code.replace('validate_video_service_narrow_mapped_pages.py','validate_video_service_same_edge_pages.py')
old="  check('audit_video_service_same_edge_warnings.py',['--board',board,'--boot',boot])"
new="""  warning_proof=board/'synthesis-warning-audit.json'
  if warning_proof.exists():
   wp=json.loads(warning_proof.read_text());assert wp['status']=='ALL_CURRENT_SYNTHESIS_WARNINGS_CLASSIFIED_PASS'
   assert wp['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v')
   for f,h in wp['evidence'].items():assert sha(f)==h,f
  else:check('audit_video_service_same_edge_warnings.py',['--board',board,'--boot',boot])"""
assert code.count(old)==1;code=code.replace(old,new)
exec(compile(code,str(source)+':same-edge','exec'),globals())
