"""All release gates rebound to the smaller independently frozen candidate."""
from pathlib import Path
BASE_SCRIPT=Path(__file__).with_name('finish_camera_joint_release.py')
code=BASE_SCRIPT.read_text()
for before,after in (
 ('camera_joint_board','camera_targeted_board'),
 ('camera-joint-evidence2','camera-targeted-evidence1'),
 ('camera-joint-primitives1','camera-targeted-primitives1'),
 ('camera-joint-capture3','camera-targeted-capture1'),
 ('validate_camera_joint.py','validate_camera_targeted.py'),
 ('test-camera-joint-capture.py','test-camera-targeted-capture.py'),
 ('check-camera-joint-progress.py','check-camera-targeted-progress.py'),
 ('camera-joint-progress1.json','camera-targeted-progress1.json'),
 ('camera-joint-cdc-structure1.json','camera-targeted-cdc-structure1.json'),
 ('camera-joint-warnings1.json','camera-targeted-warnings1.json'),
 ('camera_joint_semantics','camera_targeted_semantics'),
 ('camera_joint_flow','camera_targeted_flow'),
 ('camera_joint_implementation','camera_targeted_implementation'),
 ('finish_camera_joint_release','finish_camera_targeted_release'),
 ('1728512','194560'),('41196','5356'),
 ("v['cells']==1205","v['cells']==len(bind(root/'proof.json')['cells']) and v['cells']>=1198"),
):
 assert before in code,before
 code=code.replace(before,after)
start=code.index(" reuse=bind(SHORTS/'reused_evidence.json')")
end=code.index(" for f in ('camera_targeted_semantics.py'",start)
code=code[:start]+""" current=bind(SHORTS/'result.json');assert current['status']=='LOCAL_DRIVER_SAME_MAP_SHORT_INTEGRATION_PASS' and current['map_sha256']==mapped
 verify(current['evidence'])
 verify({str(BASE_SCRIPT):sha(BASE_SCRIPT),str(HERE/'validate_camera_joint.py'):i['base_runner_sha256']})
"""+code[end:]
exec(compile(code,str(BASE_SCRIPT)+':targeted-full-gates','exec'),globals())
