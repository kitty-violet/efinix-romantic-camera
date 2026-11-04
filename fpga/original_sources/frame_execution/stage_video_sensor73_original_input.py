"""Synthesize from the released input; reuse only identical sensor pin evidence."""
from pathlib import Path
source=Path(__file__).with_name('stage_video_sensor_rom_map.py')
code=source.read_text(encoding='utf8')
old="assert pr['inputs_sha256']==sha(pins/'inputs.json') and pi['prepared_sha256']==sha(n/'inputs.json')"
new="""assert pr['inputs_sha256']==sha(pins/'inputs.json')
 tested_native=Path(pi['prepared']);assert pi['prepared_sha256']==sha(tested_native/'inputs.json')
 same_files=['mapped_camera.v','expected.hex','tb.sv']+[k.lower()+'.v' for k in proof['kinds']]
 identical={name:sha(n/name) for name in same_files}
 assert all(sha(tested_native/name)==h for name,h in identical.items())
 parent_input=ROOT/'tmp/frame-execution/connected-board-video-service-probe60-input-isp-explicit/native_source.v'
 assert proof['original_synthesis_input_sha256']==sha(parent_input)
 assert proof['script_sha256']==sha(HERE/'prepare_video_sensor73_original_input.py')
 assert proof['base_generator_sha256']==sha(HERE/'prepare_video_sensor_rom_verify.py')"""
assert code.count(old)==1;code=code.replace(old,new)
old="builder_sha256=sha(__file__),original_sdc_unchanged=True"
new="builder_sha256=sha(__file__),base_builder_sha256=sha(source),identical_tested_sensor_files=identical,reused_pin_native=str(tested_native),original_synthesis_input_sha256=sha(parent_input),original_sdc_unchanged=True"
assert code.count(old)==1;code=code.replace(old,new)
exec(compile(code,str(source)+':original-input','exec'),globals())
