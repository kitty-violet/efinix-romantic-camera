"""Five current mapped87 sensor pin/readback/negative cases, bounded per case."""
from pathlib import Path
source=Path(__file__).with_name('test_video_sensor72.py');code=source.read_text()
assert code.count('sensor72-map-audit.json')==1
code=code.replace('sensor72-map-audit.json','control87-candidate-audit.json').replace('SENSOR72_MAPPED_I2C_PINS_PASS','CONTROL87_MAPPED_I2C_PINS_PASS')
code=code.replace('runner_sha256=sha(__file__),hardware_access=False)','runner_sha256=sha(__file__),base_runner_sha256=sha(source),hardware_access=False)')
exec(compile(code,str(source)+':current87-sensor82-values','exec'),globals())
