"""Current mapped startup/adapter reset proof includes the new sequencer."""
from pathlib import Path
from control_command_repair import once
SOURCE=Path(__file__).with_name('validate_control_adapter_idle_mapped.py')
code=SOURCE.read_text(encoding='utf8')
marker="    code=code.replace(r'\\cpu/sequencer/state ',r'\\cpu/boot_observer_state ')"
owned="def owned(n):\n    if n.startswith(('\\\\preview_boot/','\\\\preview_frame_budget[')):return True"
# There are two generated Python layers. repr keeps the embedded newline a
# literal until the extractor itself is generated; compile each layer before IO.
inner="    exec(compile(code,str(old),'exec')"
inject=f"    code=code.replace('def owned(n):', {owned!r})\n"
ports="    ports=ex['inputs'];clk=ports['process_clk_150'];reset=ports['i_arstn']"
assertion="\n    assert not any('preview_boot/' in n or 'preview_frame_budget' in n for n in ports), 'New controller state cannot be replaced by a stimulus cut'"
addition=f"\n    code=code.replace({inner!r},{inject!r}+{inner!r})\n    code=code.replace({ports!r},{(ports+assertion)!r})\n"
code=once(code,marker,marker+addition)
exec(compile(code,str(SOURCE)+':continuous-controller','exec'),globals())
