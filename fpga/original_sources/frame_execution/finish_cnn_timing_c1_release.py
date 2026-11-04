"""Same complete release checks rebound to the reviewed C1 placement origin."""
from pathlib import Path
SOURCE=Path(__file__).with_name('finish_cnn_timing_closure_release.py')
code=SOURCE.read_text()
code=code.replace('finish_cnn_timing_closure_release','finish_cnn_timing_c1_release')
code=code.replace('cnn_timing_closure_place','cnn_timing_c1_place').replace('cnn_timing_closure_route','cnn_timing_c1_route')
before='"dict(before=\'CONGESTION_2\',after=\'CONGESTION_2\')"'
after='"dict(before=\'CONGESTION_1\',after=\'CONGESTION_2\')"'
assert code.count(before)==1;code=code.replace(before,after)
exec(compile(code,str(SOURCE)+':c1-layout-release','exec'),globals())
