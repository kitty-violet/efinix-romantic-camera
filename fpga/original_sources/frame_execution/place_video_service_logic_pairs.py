"""Unchanged bounded physical flow, including both LOGIC outputs in review."""
import json, runpy, sys
from pathlib import Path
from run_units import sha
import review_existing_routability

assert 'EFX_COMB4' not in review_existing_routability.OUTPUTS
review_existing_routability.OUTPUTS['EFX_COMB4']=('O','P','CO','FCO')
source=Path(__file__).with_name('place_video_service_epoch.py')
runpy.run_path(str(source),run_name='__main__')
root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
if (root/'placement_review_inputs.json').exists():
    p=root/'placement_review_inputs.json';record=json.loads(p.read_text())
    record['logic_pair_output_review']=dict(outputs=['O','P','CO','FCO'],adapter_sha256=sha(__file__),physical_flow_changed=False)
    p.write_text(json.dumps(record,indent=2))
