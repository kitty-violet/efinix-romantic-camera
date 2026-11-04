"""All existing59 host tests plus9 version5 tests, without USB."""
from pathlib import Path
source=Path(__file__).with_name('validate_video_service_host.py');code=source.read_text()
code=code.replace("'test_video_service_compact']","'test_video_service_compact','test_video_service_narrow']")
code=code.replace('Ran 59 tests','Ran 68 tests')
exec(compile(code,str(source)+':narrow-schema','exec'),globals())
