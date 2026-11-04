"""Official isolated map + unchanged pixel/CRC tests for the compact experiment."""
import argparse, json, sys
from pathlib import Path
from run_units import ROOT, HERE, sha

p=argparse.ArgumentParser()
p.add_argument('--source-root',type=Path,required=True)
p.add_argument('--root',type=Path,required=True)
a=p.parse_args()
experiment=a.source_root.resolve()
assert experiment.is_relative_to(ROOT/'tmp')
proof=json.loads((experiment/'edit.json').read_text())
assert sha(experiment/'video_input_lut.sv')==proof['candidate_sha256']
base=HERE/'validate_video_input_mapped.py'
code=base.read_text()
old="source=HERE/'variants/video_service/video_input_lut.sv'"
assert code.count(old)==1
code=code.replace(old,'source=Path('+repr(str(experiment/'video_input_lut.sv'))+')')
# Also exercise an incorrect low half before the already tested incorrect high
# half. A commit may only accept both halves of the same completed CRC.
needle="  (root/'tb.sv').write_text(tb)"
assert code.count(needle)==1
addition='''  old="cmd(32'h49800000|crc[15:0]);cmd(32'h49c00000|(crc[31:16]^1));"
  new="cmd(32'h49800000|(crc[15:0]^1));cmd(32'h49c00000|crc[31:16]);if(tv)$fatal(1,\\"BAD_LOW_CRC_ACCEPTED\\");cmd(32'h49800000|crc[15:0]);cmd(32'h49c00000|(crc[31:16]^1));"
  assert tb.count(old)==1
  tb=tb.replace(old,new)
'''
code=code.replace(needle,addition+needle)
root=a.root.resolve()
sys.argv=[str(base),'--root',str(root)]
try:
    exec(compile(code,str(base)+':compact-isolated-map','exec'),globals())
finally:
    if root.exists():
        (root/'adapter-evidence.json').write_text(json.dumps(dict(
            adapter=str(Path(__file__)),adapter_sha256=sha(__file__),
            mapped_runner=str(base),mapped_runner_sha256=sha(base),
            experiment=str(experiment),edit_sha256=sha(experiment/'edit.json'),
            hardware_access=False,full_design_release=False),indent=2),encoding='utf8')
