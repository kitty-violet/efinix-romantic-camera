"""The existing bounded APB read worker with OFP4 fallback, no new commands."""
from pathlib import Path
_source=Path(__file__).with_name('read_command_progress_status.py')
_code=_source.read_text().replace('from read_control_progress import fresh',
                                'from read_fault_probe import fresh').replace('OFP2','OFP4')
exec(compile(_code,str(_source),'exec'),globals())
