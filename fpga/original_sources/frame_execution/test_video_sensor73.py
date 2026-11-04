"""All five affected pin checks on the new official sensor73 map."""
from pathlib import Path
source=Path(__file__).with_name('test_video_sensor72.py')
code=source.read_text(encoding='utf8')
assert code.count('sensor72-map-audit.json')==1
code=code.replace('sensor72-map-audit.json','sensor73-map-audit.json').replace('SENSOR72_MAPPED_I2C_PINS_PASS','SENSOR73_MAPPED_I2C_PINS_PASS')
needle='runner_sha256=sha(__file__),hardware_access=False)'
assert code.count(needle)==1
code=code.replace(needle,'runner_sha256=sha(__file__),base_runner_sha256=sha(source),hardware_access=False)')
exec(compile(code,str(source)+':rom-backed-verification','exec'),globals())
