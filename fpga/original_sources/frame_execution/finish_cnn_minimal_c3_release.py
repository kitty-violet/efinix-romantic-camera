"""Full existing release gates, binding the C3-specific placement and route."""
from pathlib import Path
from control_command_repair import once
source=Path(__file__).with_name('finish_cnn_minimal_free_release.py')
body=source.read_text(encoding='utf8')
for before,after in (
    ('finish_cnn_minimal_free_release','finish_cnn_minimal_c3_release'),
    ('cnn_minimal_free_route','cnn_minimal_c3_route'),
    ('cnn_minimal_repack.py','cnn_minimal_c3_place.py')):
    assert before in body
    body=body.replace(before,after)
body=once(body,"code=code_for('minimal')","code=code_for('minimal')\ncode=once(code,\"dict(before='CONGESTION_2',after='CONGESTION_2')\",\"dict(before='CONGESTION_3',after='CONGESTION_3')\")")
exec(compile(body,str(source)+':C3-complete-gates','exec'),globals())
