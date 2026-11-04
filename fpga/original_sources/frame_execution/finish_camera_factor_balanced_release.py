"""Full inherited release gates and current-map late-control evidence."""
from pathlib import Path
BASE_SCRIPT=Path(__file__).with_name('finish_camera_joint_release.py')
code=BASE_SCRIPT.read_text()
for before,after in (
 ('camera_joint_board','camera_factor_balanced_board'),
 ('camera-joint-evidence2','camera-factor-balanced-evidence1'),
 ('camera-joint-primitives1','camera-factor-balanced-primitives1'),
 ('camera-joint-capture3','camera-factor-balanced-capture1'),
 ('validate_camera_joint.py','validate_camera_factor_balanced.py'),
 ('test-camera-joint-capture.py','test-camera-factor-balanced-capture.py'),
 ('check-camera-joint-progress.py','check-camera-factor-balanced-progress.py'),
 ('camera-joint-progress1.json','camera-factor-balanced-progress1.json'),
 ('camera-joint-cdc-structure1.json','camera-factor-balanced-cdc-structure1.json'),
 ('camera-joint-warnings1.json','camera-factor-balanced-warnings1.json'),
 ('camera_joint_semantics','camera_factor_balanced_semantics'),
 ('camera_joint_flow','camera_factor_balanced_flow'),
 ('camera_joint_implementation','camera_factor_balanced_implementation'),
 ('finish_camera_joint_release','finish_camera_factor_balanced_release'),
 ("v['status']=='CAMERA_JOINT_ACTUAL_MAP_PRIMITIVES_PASS'","v['status']=='FACTOR_ACTUAL_MAP_CONTROLS_AND_DECODER_PASS'"),
 ("v['replica_checks']==1728512","v['control_checks']==786432 and v['xz_checks']>0"),
 ('1728512','786432'),('41196','5356'),
 ("v['cells']==1205","v['cells']==len(bind(root/'proof.json')['cells']) and v['cells']>=1198"),
 ('identical local drivers','exact late-control factorization'),
 ('identical local data/enable drivers','group-preserving late-control factorization'),
 ('CAMERA_JOINT_RELEASE.md','CAMERA_FACTOR_RELEASE.md'),
):
 assert before in code,before
 code=code.replace(before,after)
start=code.index(" reuse=bind(SHORTS/'reused_evidence.json')")
end=code.index(" for f in ('camera_factor_balanced_semantics.py'",start)
code=code[:start]+""" current=bind(SHORTS/'result.json');assert current['status']=='LOCAL_DRIVER_SAME_MAP_SHORT_INTEGRATION_PASS' and current['map_sha256']==mapped
 verify(current['evidence'])
 verify({str(BASE_SCRIPT):sha(BASE_SCRIPT),str(HERE/'validate_camera_factor.py'):i['base_runner_sha256']})
 load=bind(ROOT/'tmp/board-test/camera-factor-balanced-loads1.json');assert load['clock_reset_pins_equal'] and load['map_sha256']==mapped
 verify(load['evidence'])
 groups=bind(ROOT/'tmp/board-test/camera-factor-balanced-groups1.json');assert groups['status']=='ACTUAL_MAP_ALL_96_ORIGINAL_CE_GROUPS_PRESERVED' and groups['groups']==96 and groups['map_sha256']==mapped
 verify(groups['evidence'])
 depth=bind(ROOT/'tmp/board-test/camera-factor-balanced-depth1.json');assert depth['status']=='FACTOR_NO_HIDDEN_ADD_OR_SRL_ADDRESS_LATE_PATH' and depth['map_sha256']==mapped
 assert all(row['after_max']==1 for row in depth['rows']);verify(depth['evidence'])
"""+code[end:]
exec(compile(code,str(BASE_SCRIPT)+':factor-all-gates','exec'),globals())
