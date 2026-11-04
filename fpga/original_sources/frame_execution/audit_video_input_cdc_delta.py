"""Close classification of the new single-clock RAMs, without claiming STA."""
import argparse
import json
from pathlib import Path
from run_units import ROOT, sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--board', type=Path, required=True)
    a = p.parse_args()
    board = a.board.resolve()
    prior = ROOT/'tmp/frame-execution/connected-board-video-service-probe58-color-exposure'
    files = [board/'service-cdc-inventory-expanded.json', prior/'service-cdc-inventory-expanded.json',
             board/'input-isp-map-audit.json', board/'native_source.v', prior/'native_source.v', Path(__file__)]
    current, parent = [json.loads(f.read_text()) for f in files[:2]]
    assert not current['unclassified']
    assert current['cross_domain_cones'] == parent['cross_domain_cones']
    reset = 'u_ddr3_top/controller/u_efx_ddr3_soft_controller/genblk1.inst_ddr3_axi/inst_efx_ddr3/top_mc/clk_rstn'
    # The RAMs introduce no additional asynchronous data source. This is the
    # same retained service reset feeding the old frame owner and controller.
    for root in (board, prior):
        src = (root/'native_source.v').read_text()
        assert 'assign service_reset=!\\'+reset+' ;' in src
    assert len(current['isp_single_clock_ram']) == 3
    for ram in current['isp_single_clock_ram']:
        assert ram['clock'] == 'process_clk_150' and ram['inspected_source_paths'] > 0
        assert ram['external_inputs']
        assert all(x['source'] == reset and x['bit'] is None and x['pin'] in ('RE','RADDR')
                   for x in ram['external_inputs'])
    delta = json.loads(files[2].read_text())
    assert delta['status'] == 'INPUT_ISP_SOURCE_AND_RETAINED_MAP_PASS'
    assert delta['map_sha256'] == current['map_sha256']
    report = dict(status='INPUT_ISP_CDC_CLASSIFICATION_PASS_NOT_ROUTED_RELEASE',
                  map_sha256=current['map_sha256'], unchanged_crossings=6, added_single_clock_ram=3,
                  external_reset=reset, reset_changed=False, routed_evidence_still_required=True,
                  evidence={str(f.resolve()):sha(f) for f in files})
    (board/'input-isp-cdc-delta.json').write_text(json.dumps(report,indent=2))
    print(report['status'])


if __name__ == '__main__':
    main()
