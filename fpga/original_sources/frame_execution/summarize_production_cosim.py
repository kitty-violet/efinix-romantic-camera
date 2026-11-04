"""Bind separate native/reference timings to the same active fixture/proof."""
import argparse
import json
from pathlib import Path
from run_units import ROOT, sha


def read(root,status):
    r=json.loads((root/'result.json').read_text());assert r['status']==status
    assert sha(root/'simulation.log')==r['log_sha256']
    assert sha(root/'tb.sv')==r['testbench_sha256']
    assert sha(root/'active_payload.json')==r['payload_sha256']
    return r


def main():
    p=argparse.ArgumentParser()
    for x in ('native','reference','equivalence','output'):p.add_argument('--'+x,type=Path,required=True)
    a=p.parse_args();assert a.output.resolve().is_relative_to(ROOT/'tmp')
    n=read(a.native,'PRODUCTION_ACTIVE_NATIVE_SMOKE_PASS')
    r=read(a.reference,'PRODUCTION_ACTIVE_RTL_SMOKE_PASS')
    e=read(a.equivalence,'PRODUCTION_ACTIVE_DPI_EXTERNAL_EQUIVALENCE_PASS')
    assert n['native_inputs_sha256']==r['native_inputs_sha256']==e['native_inputs_sha256']
    assert n['payload_sha256']==r['payload_sha256']==e['payload_sha256']
    assert n['line']==r['line']
    nt=(a.native/'tb.sv').read_text().replace(a.native.resolve().as_posix(),'FIXTURE_ROOT')
    rt=(a.reference/'tb.sv').read_text().replace(a.reference.resolve().as_posix(),'FIXTURE_ROOT')
    assert nt==rt,'Timing fixtures must match exactly after artifact-root normalization'
    result=dict(status='ACTIVE_CNN_EXTERNAL_AXI_SAME_FIXTURE_TIMING_PASS',
        reference_seconds=r['wall_seconds'],native_dpi_seconds=n['wall_seconds'],
        host_speed_ratio=r['wall_seconds']/n['wall_seconds'],equivalence_line=e['line'],
        native_inputs_sha256=n['native_inputs_sha256'],payload_sha256=n['payload_sha256'],
        artifacts={str(x):sha(x/'result.json') for x in [a.native,a.reference,a.equivalence]},
        complete_frame=False,real_phy=False,full_board_speed_measured=False,
        scope='real production prefix through first store with external behavioral AXI and random backpressure')
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
