"""Validate optimization-only setup pressure on the immutable84 route."""
from pathlib import Path

source = Path(__file__).with_name('validate_setup_margin.py')
code = source.read_text(encoding='utf8')
old = 'from windows_process_tree import stop_launched_tree'
assert code.count(old) == 1
code = code.replace(old, 'from windows_process_tree_safe import stop_launched_tree')
exec(compile(code, str(source) + ':safe-process-tree', 'exec'), globals())
