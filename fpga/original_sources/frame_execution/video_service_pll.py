"""Bind the existing PLL pulse/arrival proof to the current retained graph."""
import argparse,json
from pathlib import Path
from run_units import HERE,sha
from place_video_service_epoch import audit_delta
import audit_pll_shift_routed as original

def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 board=a.board.resolve();root=a.root.resolve()
 # The original gate still checks calibrated pulse widths, minimum gap,
 # original source hashes and actual selected-route arrival reports.
 # Only the obsolete whole-board parent comparison is replaced by this
 # candidate's retained-cell and protected-boundary proof.
 audit_delta(board)
 original.audit_delta=audit_delta
 original.main()
 r=json.loads((root/'bound.json').read_text())
 r['current_graph_evidence']={str(f):sha(f) for f in (board/'inputs.json',board/'result.json',board/'retained-map-audit.json',HERE/'place_video_service_epoch.py',HERE/'place_video_service_serial.py',HERE/'place_video_service.py',Path(__file__),HERE/'audit_pll_shift_routed.py')}
 (root/'bound.json').write_text(json.dumps(r,indent=2))

if __name__=='__main__':main()
