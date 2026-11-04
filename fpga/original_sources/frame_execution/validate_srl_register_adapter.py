"""Same complete startup/adapter/reset test, with safe owned timeout cleanup."""
from pathlib import Path
import windows_process_tree as cleanup
from windows_process_tree_safe import stop_launched_tree
cleanup.stop_launched_tree=stop_launched_tree
source=Path(__file__).with_name('validate_continuous_adapter.py')
exec(compile(source.read_text(encoding='utf8'),str(source)+':srl-register-adapter','exec'),globals())
