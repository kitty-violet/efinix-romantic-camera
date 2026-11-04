"""Unchanged CDC checks with an explicit larger Boolean enumeration budget.

Native reimport exposes eleven leaves in frame_end/CE rather than the previous
ten-leaf cut. Enumerate the entire function; no timing/reset check is removed.
"""
from pathlib import Path
source=Path(__file__).with_name('camera_recovery_cdc.py')
code=source.read_text();needle='assert len(tokens)<=10,(sn,tokens)';assert code.count(needle)==1
code=code.replace(needle,'assert len(tokens)<=12,(sn,tokens)')
exec(compile(code,str(source)+':joint-complete-truth','exec'),globals())
