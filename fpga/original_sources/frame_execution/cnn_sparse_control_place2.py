"""Recover an externally interrupted placement; same map, seed and settings."""
from pathlib import Path
source=Path(__file__).with_name('cnn_sparse_control_place.py')
body=source.read_text(encoding='utf8')
assert body.count("'cnn-sparse-control-place1'")==1
body=body.replace("'cnn-sparse-control-place1'","'cnn-sparse-control-place2'")
exec(compile(body,str(source)+':interrupted-placement-recovery','exec'),globals())
