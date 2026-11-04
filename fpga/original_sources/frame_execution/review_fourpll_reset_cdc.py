"""Bind current reset/CDC evidence for read-only supervised diagnosis only."""
import argparse,json,re
from pathlib import Path
from run_units import ROOT,HERE,sha
from prepare_local_reset_board import audit_delta,BASE
from fourpll_reset_cdc import inventory


def review(board,batch,gray,output):
    assert not output.exists();evidence={}
    def bind(p):
        evidence[str(p)]=sha(p)
        return json.loads(p.read_text())
    def file(p):evidence[str(p)]=sha(p)
    delta=audit_delta(board)
    # Use the actual evidence chain of the prior measured-pinmap release,
    # not a guessed location for an older bounded-commissioning review.
    parent=bind(ROOT/'output/deployment/board-measured-pinmap-20261009/manifest.json')
    previous_reviews=[]
    for path,digest in parent['required_evidence'].items():
        assert sha(path)==digest
        evidence[path]=digest
        if Path(path).suffix=='.json':
            data=json.loads(Path(path).read_text(encoding='utf-8'))
            if isinstance(data,dict) and data.get('status')=='CDC_RESET_REVIEW_COMPLETE_FOR_BOUNDED_BRINGUP':previous_reviews.append(path)
    assert previous_reviews,'Missing inherited functional CDC/reset review'
    implementation=bind(batch/'reset_implementation.json')
    assert implementation['status']=='FULL_MAPPED_FOURPLL_RESET_AND_PERIPHERY_PASS'
    assert implementation['board']==str(board)
    for path,digest in implementation['required_evidence'].items():assert sha(path)==digest
    evidence.update(implementation['required_evidence'])
    source=board/'assembly/board_startup_reset.sv'
    shortroot=ROOT/'tmp/frame-execution/board-startup-reset-short2'
    short=bind(shortroot/'result.json');sm=bind(shortroot/'inputs.json')
    assert short['status']=='RESET_AND_ACTUAL_A_STORE_FOURSTATE_PASS'
    assert short['inputs_sha256']==sha(shortroot/'inputs.json')
    for row in sm['inputs']:
        assert sha(row['source'])==sha(row['snapshot'])==row['sha256']
        file(Path(row['snapshot']));file(Path(row['source']))
        name=Path(row['snapshot']).name
        if (board/'assembly'/name).exists():assert sha(board/'assembly'/name)==row['sha256']
    for row in short['rows']:
        log=shortroot/(row['top']+'.log')
        assert sha(log)==row['log_sha256'];file(log)
    binding=board/'local-reset-short'
    b=bind(binding/'result.json');bi=bind(binding/'inputs.json')
    assert b['status']=='LOCAL_RESET_EXACT_TOP_BINDING_PASS' and b['inputs_sha256']==sha(binding/'inputs.json')
    assert sha(board/'rtl/ti60f225_oob_top.v')==b['top_sha256']==bi['top_sha256']
    assert sha(source)==b['module_sha256']==bi['module_sha256']
    assert sha(binding/'simulation.log')==b['simulation_sha256'];file(binding/'simulation.log')
    mappedroot=ROOT/'tmp/frame-execution/board-startup-reset-mapped1'
    m=bind(mappedroot/'mapped_review.json');mi=bind(mappedroot/'inputs.json')
    assert m['status']=='EFINITY_MAPPED_STARTUP_RESET_FOURSTATE_PASS'
    assert m['inputs_sha256']==sha(mappedroot/'inputs.json') and m['source_sha256']==sha(source)
    for name,key in [('outflow/board_startup_reset.map.v','map_sha256'),('simulation.log','simulation_sha256')]:
        assert sha(mappedroot/name)==m[key];file(mappedroot/name)
    g=bind(gray/'result.json');gm=bind(gray/'inputs.json');gi=bind(gray/'clock_crossing_inventory.json')
    assert g['status']=='NEW_ROUTE_GRAY_SKEW_PASS_NOT_FULL_CDC_RELEASE'
    assert g['inputs_sha256']==sha(gray/'inputs.json') and g['inventory_sha256']==sha(gray/'clock_crossing_inventory.json')
    # The inventory's in-memory (start, end) pin tuples serialize as JSON lists.
    # Compare the exact persisted representation, without dropping any fields.
    assert gi==json.loads(json.dumps(inventory(gray))),'Regenerated CDC inventory differs'
    assert g['buses']==10 and g['endpoints']==70 and g['max_skew_ns']<g['limit_ns']==2.5
    for path,digest in gm['original'].items():assert sha(path)==digest
    for item in gi['reports']:assert sha(item['path'])==item['sha256'];file(Path(item['path']))
    for crossing in gi['new_global_reset_paths']:
        assert crossing['start'].startswith('startup_reset/released~FF|')
        assert crossing['end'].startswith('u_ddr3_top/controller/') and crossing['end'].endswith('~FF|SR')
    # All public functional logic except the explicitly audited reset producer
    # is unchanged. Vendor reset synchronizers/DDR configuration are retained.
    for name in ('ddr516_reference.sv','ddr516_v4_controller.sv','ddr516_rx_retime.sv','ddr3_controller.bin'):
        assert sha(board/name)==sha(BASE/name);file(board/name)
    vendor=(board/'ddr516_reference.sv').read_text()
    assert 'efx_resetsync_wr_rst' in vendor and 'efx_resetsync_rd_rst' in vendor
    file(Path(__file__));file(HERE/'fourpll_reset_cdc.py')
    result=dict(status='FOURPLL_RESET_CDC_REVIEW_FOR_READ_ONLY_DIAGNOSTICS',board=str(board),gray=str(gray),
        delta=delta,unknown_crossings=0,required_evidence=evidence,previous_reviews=previous_reviews,
        reviewed_classes=gi['classes'],new_global_reset_paths=gi['new_global_reset_paths'],
        findings=[
          'Direct key/LOCKED assertion; two FF lock qualification plus 64 local cycles separately in process150 and CLK5M; full mapped truth table checked.',
          'Process-domain functional reset release is clocked. New process-to-DDR endpoints accepted here are reset pins only, never payload/control data.',
          'Unchanged vendor clock-specific reset interfaces and reset synchronizers retained; no encrypted calibration edits.',
          'Read-only status passes through 64 actual FFs in two stages; multi-bit samples are non-atomic and never authorize functional traffic.',
          'Current routed Gray checks cover all 10 groups/70 endpoints; setup/hold acceptance remains the original audited clock relationships.',
          'Only temporary configuration and idle register reads are released. Actual calibration and idle error clearance must be observed before any workload.'],
        conditions=['User present; fresh USB enumeration and exact frozen bit/hash.',
                    'Wait at least one second after configuration; all keys released; then bounded repeated reads.',
                    'If signature, loader error, or calibration is abnormal, stop; no DDR payload, weight load or CNN run.'],
        ddr_workload_released=False,full_functional_reset_proof=False,analogue_metastability_proven=False,
        encrypted_vendor_internal_cdc_proof=False,board_tested=False)
    output.write_text(json.dumps(result,indent=2));print(result['status'])


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('board','batch','gray','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();review(**{k:v.resolve() for k,v in vars(a).items()})
