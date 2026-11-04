"""Use a verified, analysis-only LUT view of actual mapped LOGIC pairs.

The synthesis/VDB/physical inputs stay untouched. Splitting the two outputs
for cone extraction avoids treating the unused I2-to-P dependency as a real
input to the boot cone. The original actual-map hash remains the evidence key.
"""
import json, os, sys
from pathlib import Path
import run_units
from map_video_service_safe import safe_run
from native_logic_split import split_logic_comb

run_units.run=safe_run
run_units.MS=Path('E:/quatra/questa_fse/win64')
os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
source=Path(__file__).with_name('validate_video_service_mapped_boot.py')
code=source.read_text()
needle="src=board/'outflow/ti60f225_oob.map.v';text=src.read_text();inputs={str(src):sha(src),str(Path(__file__)):sha(__file__)}"
replacement=needle+"""
  from native_logic_split import split_logic_comb
  original_map=text
  text,logic_pair_rows=split_logic_comb(text)
  native=Path(json.loads((board/'inputs.json').read_text())['native'])
  expected=json.loads((native/'graft.json').read_text())['logic_pairs']['pairs']
  assert {r['name'].lstrip('\\\\') for r in logic_pair_rows}=={r['pair_name'] for r in expected}
  proof=json.loads((board/'retained-map-audit.json').read_text())
  assert proof['status']=='VIDEO_SERVICE_RETAINED_MAPPED_GRAPH_PASS' and proof['map_sha256']==sha(src)
  inputs.update({str(p):sha(p) for p in [board/'retained-map-audit.json',native/'native-graph-audit.json',HERE/'native_logic_split.py',HERE/'run_video_service_logic_pair_boot.py']})
  (out/'analysis-only-logic-pair-projection.json').write_text(json.dumps(dict(original_map_sha256=sha(src),pairs=logic_pair_rows,physical_inputs_changed=False),indent=2))
"""
assert code.count(needle)==1;code=code.replace(needle,replacement)
code=code.replace("cache=board/'service-validation-graph.pkl'","cache=out/'analysis-view-graph.pkl'")
assert "if kind not in ('EFX_LUT4','EFX_ADD')" in code
exec(compile(code,str(source)+':analysis-only-logic-pair-projection','exec'),globals())
