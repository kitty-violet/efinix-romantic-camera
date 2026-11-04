"""Reuse actual mapped writer checks for each independent small candidate."""
import sys
from pathlib import Path
from control_command_repair import once

assert len(sys.argv) in (2, 3) and sys.argv[1] in ('minimal', 'data64')
variant = sys.argv[1]
attempt = int(sys.argv[2]) if len(sys.argv) == 3 else 1
assert 1 <= attempt <= 3
parent = Path(__file__).with_name('validate_cnn_timing_controls.py')
code = parent.read_text(encoding='utf-8')
code = once(code, 'cnn-timing-closure-controls1', f'cnn-{variant}-controls{attempt}')
code = once(code, 'from cnn_timing_closure_board import', f'from cnn_{variant}_board import')
code = once(code, 'Path(__file__),SOURCE,BOARD/',
            f'Path(__file__),Path({parent.as_posix()!r}),SOURCE,BOARD/')
exec(compile(code, str(parent) + ':' + variant, 'exec'), globals())
