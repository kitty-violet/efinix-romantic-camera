"""Apply the proven ROM-verification delta to the original synthesis input.

Do not feed the previous full mapped output back through synthesis: that
rewraps protected hierarchy and changes the physical packing starting point.
The extracted sensor circuit must remain byte-identical to the pin-tested one.
"""
from pathlib import Path
source=Path(__file__).with_name('prepare_video_sensor_rom_verify.py')
code=source.read_text(encoding='utf8')
edits={
 "original=MAP.read_text(encoding='utf8');words,blocks=read_rom(original)":
 "original=(PARENT/'native_source.v').read_text(encoding='utf8');words,blocks=read_rom(original)",
 "source_sha256=sha(root/'native_source.v'),script_sha256=sha(__file__),files=":
 "source_sha256=sha(root/'native_source.v'),script_sha256=sha(__file__),base_generator_sha256=sha(source),original_synthesis_input_sha256=sha(PARENT/'native_source.v'),files=",
}
for old,new in edits.items():
 assert code.count(old)==1,old
 code=code.replace(old,new)
exec(compile(code,str(source)+':original-input','exec'),globals())
