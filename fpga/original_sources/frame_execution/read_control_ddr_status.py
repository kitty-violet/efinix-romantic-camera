"""Same bounded read-only APB contract; OFP3 snapshot implementation."""
from pathlib import Path

# Preserve the proven worker timeout, read whitelist and progress accounting.
# Re-enter this module in the worker so it can never use an OFP2 scan by mistake.
_source=Path(__file__).with_name('read_command_progress_status.py')
_code=_source.read_text().replace('from read_control_progress import fresh',
                                'from read_control_ddr_progress import fresh').replace('OFP2','OFP3')
exec(compile(_code,str(_source),'exec'),globals())
