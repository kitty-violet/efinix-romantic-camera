"""Reuse the bounded official-primitive I2C tests for the minimal replacement."""
from pathlib import Path
source=Path(__file__).with_name('test_video_sensor_rom_verify.py')
code=source.read_text(encoding='utf8')
for old,new in {
    'SENSOR_ROM_VERIFY_CANDIDATE_NOT_SYNTHESIZED':'SENSOR74_MINIMAL_CANDIDATE_NOT_SYNTHESIZED',
    'prepare_video_sensor_rom_verify.py':'prepare_video_sensor74_minimal.py',
    'ROM_BACKED_SENSOR_VERIFY_PIN_PROTOTYPE_PASS_NOT_PHYSICAL_RELEASE':'SENSOR74_MINIMAL_PIN_PASS_NOT_PHYSICAL_RELEASE',
    'runner_sha256=sha(__file__),modes=modes':'runner_sha256=sha(__file__),base_runner_sha256=sha(source),modes=modes',
}.items():
    assert code.count(old)==1,old
    code=code.replace(old,new)
exec(compile(code,str(source)+':sensor74-minimal','exec'),globals())
