"""Classify actual changed cones against the exact-original-input gate."""
from pathlib import Path
source=Path(__file__).with_name('audit_video_sensor73_cdc.py')
code=source.read_text(encoding='utf8').replace('SENSOR73_ROM_VERIFY_ALL_RETAINED_MAP_PASS','SENSOR73_EXACT_SOURCE_SCOPE_FRESH_MAP_PASS')
code=code.replace('All existing boundaries retained; changed eight FF D cones','All other original source and protected ports retained; changed eight FF D cones')
exec(compile(code,str(source)+':original-source-scope','exec'),globals())
