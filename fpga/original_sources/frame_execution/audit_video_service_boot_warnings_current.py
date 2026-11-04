"""Check every current warning, without assuming the previous map's count."""
from pathlib import Path
source=Path(__file__).with_name('audit_video_service_boot_warnings.py')
code=source.read_text()
old=' assert len(warnings)==968'
assert code.count(old)==1
code=code.replace(old,' assert warnings, "Expected warnings from exact current primitive cone"')
code=code.replace("mapped,root/'simulate.log',root/'result.json',Path(__file__)","mapped,root/'simulate.log',root/'result.json',Path(__file__),source")
exec(compile(code,str(source)+':current-count','exec'),globals())
