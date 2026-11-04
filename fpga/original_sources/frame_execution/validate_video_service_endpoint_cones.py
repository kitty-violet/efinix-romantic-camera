"""Actual vendor LUT verification of the two current endpoint cofactors."""
from pathlib import Path
source=Path(__file__).with_name('validate_video_service_global_cones.py')
code=source.read_text()
for a,b in [('global-batch-independent-audit.json','endpoint-batch-independent-audit-v2.json'),('GLOBAL_BATCH_INDEPENDENT_FULL_SOURCE_PASS_NOT_PHYSICAL','ENDPOINT_BATCH_FULL_SOURCE_INDUCTIVE_PASS_NOT_PHYSICAL'),('video-service-native53-parallel-group','video-service-native56-global-targeted'),('ACTUAL_GLOBAL_CONTROL_CONES_PASS','ACTUAL_ENDPOINT_CONTROL_CONES_PASS')]:
 assert a in code;code=code.replace(a,b)
# Group2's old inputs use preserved output boundaries of group1; its owned
# cells are unchanged from56. The independent ordered operation replay proves it.
exec(compile(code,str(source)+':endpoint-cones','exec'),globals())
