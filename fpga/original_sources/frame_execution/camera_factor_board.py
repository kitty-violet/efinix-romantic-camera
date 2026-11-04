"""Full board map of exact late-control cofactor candidate."""
from pathlib import Path
BASE_SCRIPT=Path(__file__).with_name('camera_joint_board.py')
code=BASE_SCRIPT.read_text().replace('connected-board-camera-joint3','connected-board-camera-factor1').replace('camera-joint-native6','camera-factor-native2')
code=code.replace("r['status']=='JOINT_NATIVE_EXACT_TRANSFORM_PASS'","r['status']=='FACTOR_NATIVE_EXACT_TRANSFORM_PASS'")
needle='(Path(__file__),TRANSFORM/';assert code.count(needle)==1
code=code.replace(needle,'(Path(__file__),BASE_SCRIPT,TRANSFORM/')
# CE reassignment can make old combinational endpoints dead. The following
# existing audit rejects any missing cell whose output has a retained user.
# No sequential or protected cell is allowed to disappear.
old="assert all(n.startswith('JOINT_DIV_') and sg[n][0]=='EFX_LUT4' for n in missing),sorted(missing)[:20]"
assert code.count(old)==1
code=code.replace(old,"assert all(sg[n][0]=='EFX_LUT4' and not n.startswith('protected_inst/') for n in missing),sorted(missing)[:20]")
exec(compile(code,str(BASE_SCRIPT)+':factor-exact','exec'),globals())
