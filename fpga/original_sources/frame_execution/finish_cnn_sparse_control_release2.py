"""Same complete release gates, binding the replacement placement launcher."""
from pathlib import Path
source=Path(__file__).with_name('finish_cnn_sparse_control_release.py')
body=source.read_text(encoding='utf8')
body=body.replace('finish_cnn_sparse_control_release','finish_cnn_sparse_control_release2')
body=body.replace('cnn_sparse_control_place.py','cnn_sparse_control_place2.py')
exec(compile(body,str(source)+':interrupted-placement-recovery','exec'),globals())
