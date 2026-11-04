"""Full-height/narrow-row cancellation, complementing the1920-wide gate.

Not a full1920x1080 consumer simulation. The prior full attempt timed out.
"""
from pathlib import Path
from run_units import sha
source=Path(__file__).with_name('validate_raw_803_boundary.py')
code=source.read_text()
code=code.replace("('PPL=32,LINES=12','PPL=960,LINES=1080')", "('PPL=32,LINES=12','PPL=32,LINES=1080')")
code=code.replace("'wire [9:0] ux;wire [10:0] uy'", "'wire [4:0] ux;wire [10:0] uy'")
code=code.replace('camera_frame(1,385440,0,3)', 'camera_frame(1,PPL*803/2,0,3)')
code=code.replace('short_words=385440', 'short_rows=803 width_pixels=64')
code=code.replace('RAW_803_FULL_GEOMETRY', 'RAW_803_FULL_HEIGHT_NARROW')
code=code.replace('full1920x1080 geometry', '64x1080 geometry (full height only)')
code=code.replace('simulation_host_timeout_seconds=180', 'simulation_host_timeout_seconds=45')
code=code.replace("root,'simulation.log',180", "root,'simulation.log',45")
code=code.replace("runner_sha256=sha(__file__),tb_sha256", "runner_sha256=sha(__file__),adapter_source_sha256=sha(source),tb_sha256")
exec(compile(code,str(source)+':segmented', 'exec'),globals())
