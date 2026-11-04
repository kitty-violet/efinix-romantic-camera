"""Use the existing official memory-update audit on released73, without P&R."""
from pathlib import Path
source=Path(__file__).with_name('update_video_sensor_gain_bram.py')
code=source.read_text(encoding='utf8')
edits={
 "BASE = ROOT/'tmp/frame-execution/video-input-bit2'":"BASE = ROOT/'tmp/frame-execution/video-sensor73-bits9'",
 "CANDIDATE = ROOT/'tmp/frame-execution/video-sensor-gain70-candidate1'":"CANDIDATE = ROOT/'tmp/frame-execution/video-exposure76-candidate1'",
 "b9da6275178e6bf83c76ee4fbcdbb2f956aae9986b37c12a0402cb0ce535311b":"1274fef0526d75b0b3b32f7f0d72c62a77839e077c1085d8038a6c1ec6b2a562",
 "shutil.copytree(BASE,root)":"shutil.copytree(BASE,root,ignore=shutil.ignore_patterns('work_syn','before_route','same-map-geometry.json'))",
 "OFFICIAL_BRAM_GAIN_OVERLAY_PASS_NOT_RELEASED":"OFFICIAL_BRAM_EXPOSURE76_OVERLAY_PASS_NOT_RELEASED",
 "tool_sha256=sha(UPDATER),runner_sha256=sha(__file__),steps=steps,":"tool_sha256=sha(UPDATER),runner_sha256=sha(__file__),base_runner_sha256=sha(source),steps=steps,",
}
for old,new in edits.items():
 assert code.count(old)==1,old
 code=code.replace(old,new)
exec(compile(code,str(source)+':exposure76-released73','exec'),globals())
