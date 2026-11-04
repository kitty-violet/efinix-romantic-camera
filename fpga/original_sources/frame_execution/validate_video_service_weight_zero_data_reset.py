"""Current mapped data-only-reset proof for the guarded A-zero child."""
from pathlib import Path
source=Path(__file__).with_name('validate_video_service_weight_data_reset.py')
code=source.read_text();assert code.count('weight-lookahead-edit.json')==2
code=code.replace('weight-lookahead-edit.json','weight-zero-edit.json')
exec(compile(code,str(source)+':a-zero-child','exec'),globals())
