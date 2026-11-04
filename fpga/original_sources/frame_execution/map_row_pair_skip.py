"""Map the retained-RAM residual merge with its actual feedback FIFOs."""
import argparse,json
from pathlib import Path
from run_units import ROOT,HERE,sha
from map_area_screen import one
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
files=[HERE/'variants/row_pair_arena/row_pair_skip_fuse4.sv',ROOT/'tmp/frame-execution/connected-board-compact-feed1/assembly/shallow_result_fifo.sv']
source=r/'row_pair_skip_fuse4.sv';source.write_text('\n'.join(f.read_text() for f in files),encoding='utf-8')
manifest=dict(inputs=[dict(source=str(f),sha256=sha(f)) for f in files],generated_sha256=sha(source),runner_sha256=sha(__file__))
(r/'inputs.json').write_text(json.dumps(manifest,indent=2)+'\n')
result=one(r/'map',source,'speed',budget=240)
for f,rec in zip(files,manifest['inputs']):assert sha(f)==rec['sha256']
result.update(status='ROW_PAIR_SKIP_WITH_FEEDBACK_FIFO_MAP_NOT_BOARD',inputs_sha256=sha(r/'inputs.json'),whole_board=False,physical_xlr=None)
(r/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
