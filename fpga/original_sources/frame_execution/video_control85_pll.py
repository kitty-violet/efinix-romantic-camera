"""Bind the unchanged PLL protocol to actual83 routed arrival reports."""
import argparse,json
from pathlib import Path
from run_units import sha
from video_control85_physical import audit,BOARD
import audit_pll_shift_routed as original
def audit_delta(board):
 assert board.resolve()==BOARD;return audit()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 original.audit_delta=audit_delta;original.main()
 root=a.root.resolve();r=json.loads((root/'bound.json').read_text())
 r['current_graph_evidence']={str(f):sha(f) for f in (Path(__file__),BOARD/'control85-preplace-audit.json',BOARD/'inputs.json',BOARD/'result.json')}
 (root/'bound.json').write_text(json.dumps(r,indent=2))
