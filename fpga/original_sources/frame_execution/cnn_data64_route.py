"""Original constraint route/STA/Gray gates, bounded to 2700 seconds."""
from pathlib import Path
SOURCE = Path(__file__).with_name('cnn_timing_closure_route.py')
code = SOURCE.read_text(encoding='utf-8').replace('cnn_timing_closure', 'cnn_data64')
exec(compile(code, str(SOURCE) + ':data64', 'exec'), globals())
