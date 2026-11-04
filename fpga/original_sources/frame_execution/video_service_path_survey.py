"""Read-only path survey with creation-time-safe timeout cleanup."""
from pathlib import Path
source=Path(__file__).with_name('survey_routed_paths.py')
code=source.read_text()
old='from windows_process_tree import stop_launched_tree'
assert code.count(old)==1
exec(compile(code.replace(old,'from windows_process_tree_safe import stop_launched_tree'),str(source)+':safe-cleanup','exec'),globals())
