"""Use official updater; gain write and verification bytes change together."""
from pathlib import Path
source=Path(__file__).with_name('update_video_sensor_gain_bram.py')
code=source.read_text(encoding='utf8')
edits={
 "BASE = ROOT/'tmp/frame-execution/video-input-bit2'":"BASE = ROOT/'tmp/frame-execution/video-sensor73-bits9'",
 "CANDIDATE = ROOT/'tmp/frame-execution/video-sensor-gain70-candidate1'":"CANDIDATE = ROOT/'tmp/frame-execution/video-gain82-candidate1'",
 "b9da6275178e6bf83c76ee4fbcdbb2f956aae9986b37c12a0402cb0ce535311b":"1274fef0526d75b0b3b32f7f0d72c62a77839e077c1085d8038a6c1ec6b2a562",
 "shutil.copytree(BASE,root)":"shutil.copytree(BASE,root,ignore=shutil.ignore_patterns('work_syn','before_route','same-map-geometry.json'))",
 "i=int(address);assert original[i] == c['register']<<9|c['old']<<1\n        expected[i]=c['register']<<9|c['new']<<1\n        values[i]=expected[i]&0xfffff":
 "i=int(address)\n        if i<164:\n            assert original[i] == c['register']<<9|c['old']<<1\n            expected[i]=c['register']<<9|c['new']<<1\n            values[i]=expected[i]&0xfffff\n        else:\n            assert i==266 and c['kind']=='readback_expected_byte'\n            assert values[i]==c['old']<<1\n            values[i]=c['new']<<1",
 "OFFICIAL_BRAM_GAIN_OVERLAY_PASS_NOT_RELEASED":"OFFICIAL_BRAM_GAIN82_OVERLAY_PASS_NOT_RELEASED",
 "tool_sha256=sha(UPDATER),runner_sha256=sha(__file__),steps=steps,":"tool_sha256=sha(UPDATER),runner_sha256=sha(__file__),base_runner_sha256=sha(source),steps=steps,",
}
for old,new in edits.items():
 assert code.count(old)==1,old
 code=code.replace(old,new)
exec(compile(code,str(source)+':gain82','exec'),globals())
