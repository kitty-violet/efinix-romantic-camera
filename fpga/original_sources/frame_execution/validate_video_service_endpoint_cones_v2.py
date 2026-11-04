"""Same actual cone comparisons, retained-driver resolution for mapped cuts."""
from pathlib import Path
from bind_video_service_mapped_cut import bind_cut
parent=Path(__file__).with_name('validate_video_service_global_cones.py');code=parent.read_text()
for a,b in [('global-batch-independent-audit.json','endpoint-batch-independent-audit-v2.json'),('GLOBAL_BATCH_INDEPENDENT_FULL_SOURCE_PASS_NOT_PHYSICAL','ENDPOINT_BATCH_FULL_SOURCE_INDUCTIVE_PASS_NOT_PHYSICAL'),('video-service-native53-parallel-group','video-service-native56-global-targeted'),('ACTUAL_GLOBAL_CONTROL_CONES_PASS','ACTUAL_ENDPOINT_CONTROL_CONES_PASS')]:
 assert a in code;code=code.replace(a,b)
a=code.index('  if side and board:');b=code.index('  assert external<=set(bindings)',a)
code=code[:a]+'''  if side and board:
   bindings,binding_rows=bind_cut(source,new,{t:j for j,t in enumerate(boundary)},external)
''' +code[b:]
code=code.replace('paths=[Path(__file__),',"paths=[Path(__file__),Path(__file__).with_name('validate_video_service_global_cones.py'),Path(__file__).with_name('bind_video_service_mapped_cut.py'),")
exec(compile(code,str(parent)+':endpoint-retained-drivers-v2','exec'),globals())
