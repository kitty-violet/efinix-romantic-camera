"""Every current mapped cloned LUT/FF under all 256 four-state input patterns."""
from pathlib import Path
from control_command_repair import once
SOURCE=Path(__file__).with_name('validate_native_local_drivers.py')
code=SOURCE.read_text()
code=once(code,"r['status']=='FULL_NATIVE_LOCAL_DRIVERS_MAP_GRAPH_PASS'","r['status']=='CNN_TIMING_CLOSURE_FULL_BOARD_MAP_PASS'")
code=once(code,"runner_sha256=sha(__file__),files=","runner_sha256=sha(__file__),base_runner_sha256=sha(SOURCE),files=")
exec(compile(code,str(SOURCE)+':cnn-closure-drivers','exec'),globals())
