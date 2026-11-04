"""Record the explicit compensated-risk decision after current-map short gates.

This records a bounded experiment, not a routability guarantee or board release.
"""
import json
from run_units import ROOT,HERE,sha
from local_driver_board import audit_delta

def main():
    base=ROOT/'tmp/frame-execution'
    board=base/'connected-board-local-drivers1';place=base/'native-local-driver-place1'
    shorts=base/'native-local-driver-evidence3';output=base/'native-local-driver-route-review1.json'
    assert not output.exists()
    audit_delta(board)
    sr=json.loads((shorts/'result.json').read_text())
    assert sr['status']=='LOCAL_DRIVER_SAME_MAP_SHORT_INTEGRATION_PASS'
    assert sr['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v')
    assert sr['board_inputs_sha256']==sha(board/'inputs.json')
    for f,h in sr['evidence'].items():assert sha(f)==h,f
    geometry=json.loads((place/'local_driver_geometry.json').read_text())
    assert geometry['groups_count']==33 and geometry['improved_groups']==32
    for f,h in geometry['evidence'].items():assert sha(f)==h,f
    result=json.loads((place/'result.json').read_text())
    assert result['status']=='PLACEMENT_ONLY_FINISHED_NOT_RELEASE'
    evidence=dict(geometry['evidence'])
    for f in (place/'result.json',place/'physical_inputs.json',place/'placement_review_inputs.json',
              place/'local_driver_geometry.json',shorts/'result.json',
              base/'native-local-drivers-map2/result.json',base/'native-local-drivers-fourstate1/result.json',
              base/'native-split-timing-route1/selected.json',HERE/'review_local_driver_route.py'):
        evidence[str(f)]=sha(f)
    review=dict(
        decision='PROCEED_TO_BOUNDED_ROUTE',board=str(board),placement=str(place),shorts=str(shorts),
        physical=str(base/'native-local-driver-bound1'),budget_seconds=1350,
        route_profile_cli_override=dict(before='CONGESTION_2',after='TIMING_1'),
        rationale=[
            'The previous legal TIMING_1 route failed setup at -1.335ns, dominated by data interconnect. This candidate targets 33 measured slow nets and 215/387 sampled failing endpoints; it is a structural fanout change, not another seed sweep.',
            'Actual new placement reduces total Manhattan load distance on 32/33 targeted nets (median ratio0.384). The worst old n67396_2 branch drops from75 loads to19 maximum; total-distance ratio0.793. This supports measuring the routing benefit but does not predict exact slack.',
            'Compensation for worse global HPWL(+6.34%) and routing-only cells(+1132) is the measured load reduction and improved fine4x4 congestion(2.0868 to1.96784). Medium/coarse congestion is worse and remains an explicit risk; no universal congestion threshold is asserted.',
            '56632/60800 XLR,228/256RAM,160/160DSP fit the permitted absolute capacity. Ordinary placement remains free, with unchanged113CNNRAM/288DSP anchors and complete cascade topology.',
            'Added52 synchronous150MHz FF retain their original D/CE/CLK/SR and parameters;242LUT clones retain exact functions. Full public net bijection, official four-state checks and current-map control/interface/reset tests bind this candidate. Protected source, pins and DDR/HDMI clocks are unchanged.'
        ],
        remaining_risks=[
            '172 sampled negative endpoints are not directly targeted. The new layout may move those paths or expose new slow paths.',
            'feature_count[0] total load distance worsens1.123x; cloning adds upstream load and two selected source nets now have32 pin loads. The24 partition cap is per cell/port group, not a physical pin guarantee.',
            'Global wire length and routing-only resource cost grew. Placement setup estimates are not route STA; negative hold estimates still require legal routing and strict hold checks.',
            'Success requires zero route overlaps, all original clock/constraint checks, WNS>=0 and WHS>=0, nonempty70-bit Gray checks, CDC/reset/PLL and same-source final package. A timeout or failure never permits download.',
            'Board DDR calibration and control startup remain unproven. No hardware access is authorized by this offline review.'
        ],reviewed_evidence=evidence,hardware_access=False,board_release=False)
    output.write_text(json.dumps(review,indent=2),encoding='utf8')
    print(output)

if __name__=='__main__':main()
