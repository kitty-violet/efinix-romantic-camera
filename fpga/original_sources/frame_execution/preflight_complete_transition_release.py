"""Read-only candidate release-evidence preflight; no timing or board release."""
import json
from run_units import ROOT,sha
from complete_control_transition import audit_delta
import local_control_decode
local_control_decode.audit_delta=audit_delta
import audit_local_control_decode_map
audit_local_control_decode_map.audit_delta=audit_delta
from finish_complete_transition_release import PARENT,SERIAL,parent_evidence,extra_evidence

def main():
    evidence={}
    def bind(p):
        evidence[str(p)]=sha(p)
        return json.loads(p.read_text())
    def verify(rows):
        for p,h in rows.items():
            assert sha(p)==h,p
            evidence[str(p)]=h
    parent=bind(PARENT/'manifest.json');parent_evidence(parent,verify)
    verify({str(PARENT/n):h for n,h in parent['bundle_files'].items()})
    b=ROOT/'tmp/frame-execution'
    extra_evidence(b/'connected-board-complete-transition2',b/'complete-transition-route2',b/'complete-transition-place2',SERIAL,bind,verify)
    p=b/'complete-transition-release-preflight2.json';assert not p.exists()
    r=dict(status='RELEASE_EVIDENCE_PREFLIGHT_ONLY',files=len(evidence),required_evidence=evidence,
        route_sta_gray_pll_bitstream_still_required=True,hardware_access=False,board_release=False)
    p.write_text(json.dumps(r,indent=2));print(r['status'],len(evidence))
if __name__=='__main__':main()
