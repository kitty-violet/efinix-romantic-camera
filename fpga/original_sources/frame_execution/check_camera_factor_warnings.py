"""Original protected-open checks; factorization introduces no unused carry."""
from pathlib import Path
source=Path(__file__).with_name('check_camera_joint_warnings.py')
code=source.read_text().replace('from camera_joint_board import','from camera_factor_board import')
needle='assert len(explained)==21,len(explained)';assert code.count(needle)==1
code=code.replace(needle,'assert len(explained)==0,len(explained)')
exec(compile(code,str(source)+':factor-no-new-dangling','exec'),globals())
