from pathlib import Path
source=Path(__file__).with_name('camera_joint_semantics.py')
exec(compile(source.read_text().replace('camera_joint_board','camera_factor_balanced_board'),str(source)+':factor-balanced','exec'),globals())
