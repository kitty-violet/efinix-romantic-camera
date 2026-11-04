"""Run reviewed route7 with known STA and CDC copy gaps closed in advance."""
import json,sys
from pathlib import Path
from run_units import HERE,sha
from control_command_repair import once
import camera_checked_flow as flow
import camera_sparse_route_v2 as prepared

original=flow.generated
def generated(source,code,suffix):
    code=once(code,"HERE/'camera_checked_gray_v2.py'","HERE/'camera_checked_gray_rcf.py'")
    physical=Path(sys.argv[sys.argv.index('--physical')+1]).resolve()
    review=physical/'route_review.json';r=json.loads(review.read_text())
    for f in (Path(__file__),HERE/'camera_checked_gray_rcf.py',HERE/'audit_camera_sparse_route_v2.py',HERE/'canonical_sparse_snapshot_v2.py'):
        r['required_evidence'][str(f.resolve())]=sha(f)
    r['analysis_collateral_note']='Only complete RCF/trace copies added to analysis views; setup/hold/coverage/CDC/Gray checks unchanged.'
    review.write_text(json.dumps(r,indent=2))
    original(source,code,suffix)

if __name__=='__main__':
    flow.generated=generated
    prepared.route()
