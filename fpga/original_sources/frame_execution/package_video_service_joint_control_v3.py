"""Unchanged64 release gates; normalize native Windows artifact receipt paths."""
from pathlib import Path

source = Path(__file__).with_name('package_video_service_joint_control_v2.py')
code = source.read_text()
old = " bit=bits/'outflow/ti60f225_oob.bit';assert generated['artifacts']['outflow/ti60f225_oob.bit']['sha256']==sha(bit)"
new = """ bit=bits/'outflow/ti60f225_oob.bit'
 from deployment_artifact_paths import checked_artifact
 bind(HERE/'deployment_artifact_paths.py')
 bind(HERE/'package_video_service_joint_control_v2.py')
 artifact=checked_artifact(generated['artifacts'],bits,bit)
 assert artifact['sha256']==sha(bit) and artifact['bytes']==bit.stat().st_size"""
anchor = 'anchor="source=Path(__file__).with_name'
assert code.count(anchor) == 1
code = code.replace(anchor, 'scope_changes.append(' + repr((old, new)) + ')\n' + anchor)
exec(compile(code, str(source) + ':windows-artifact-binding', 'exec'), globals())
