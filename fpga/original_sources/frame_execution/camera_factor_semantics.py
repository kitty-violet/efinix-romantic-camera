"""Compose the verified parent and current reimport net names."""
from pathlib import Path
source=Path(__file__).with_name('camera_joint_semantics.py')
exec(compile(source.read_text().replace('camera_joint_board','camera_factor_board'),str(source)+':factor','exec'),globals())
