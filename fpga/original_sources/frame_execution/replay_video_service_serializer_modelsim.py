"""Identical unencrypted fixture on installed ModelSim, no option suppression."""
from pathlib import Path
source=Path(__file__).with_name('rerun_video_service_serializer.py');code=source.read_text()
old="sim=Path('E:/quatra/questa_fse/win64')"
assert code.count(old)==1
code=code.replace(old,"sim=Path('E:/altera/15.0/modelsim_ase/win32aloem')")
code=code.replace('runner_sha256=sha(__file__),','runner_sha256=sha(__file__),original_runner_sha256=sha(source),')
exec(compile(code,str(source)+':modelsim-unencrypted','exec'),globals())
