"""Same complete offline flows, bound to the new exact-ready map only."""
from pathlib import Path
source=Path(__file__).with_name('camera_payload_flow.py')
code=source.read_text(encoding='utf8').replace('camera_payload_board','camera_ready_board').replace('camera_payload_implementation','camera_ready_implementation').replace('camera_payload_flow','camera_ready_flow')
exec(compile(code,str(source)+':ready-bound', 'exec'),globals())
