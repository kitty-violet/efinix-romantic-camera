"""Read-only release evidence preflight while routing; never grant release."""
import json
from pathlib import Path
from run_units import ROOT,sha
from accepted_tail_board import audit_delta
import local_control_decode
local_control_decode.audit_delta=audit_delta
import audit_local_control_decode_map
audit_local_control_decode_map.audit_delta=audit_delta
from finish_accepted_tail_release import extra_evidence,parent_evidence,SERIAL

def main():
 evidence={}
 def bind(p):evidence[str(p)]=sha(p);return json.loads(Path(p).read_text())
 def verify(rows):
  for p,h in rows.items():assert sha(p)==h,p;evidence[str(p)]=h
 parent=ROOT/'output/deployment/board-full-150-command-progress-20261010'
 m=bind(parent/'manifest.json');parent_evidence(m,verify);verify({str(parent/n):h for n,h in m['bundle_files'].items()})
 base=ROOT/'tmp/frame-execution';board=base/'connected-board-accepted-tail2';batch=base/'accepted-tail-route1';physical=base/'accepted-tail-place1'
 extra_evidence(board,batch,physical,SERIAL,bind,verify)
 result=dict(status='RELEASE_EVIDENCE_PREFLIGHT_ONLY',files=len(evidence),required_evidence=evidence,
  final_route_sta_gray_pll_bitstream_pending=True,hardware_access=False,board_release=False)
 out=base/'accepted-tail-release-preflight1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2));print(result['status'],len(evidence))
if __name__=='__main__':main()
