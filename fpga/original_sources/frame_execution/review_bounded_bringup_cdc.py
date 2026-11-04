"""Bind reviewed CDC/reset boundaries to this route for bounded commissioning.

This is an engineering boundary review, not a claim of analog metastability proof
or an exhaustive proof of encrypted vendor internals. No constraints are edited.
"""
import argparse
from pathlib import Path
from accepted_a_assets import ROOT, read_json, write_json, sha
from audit_board_bringup_evidence import receipt


def review(output):
    board=ROOT/'tmp/frame-execution/connected-board-skip-credit1'
    evidence={}
    def bind(p):
        p=Path(p);evidence[str(p)]=sha(p);return read_json(p)
    bm=bind(board/'inputs.json');br=bind(board/'result.json')
    assert sha(board/'inputs.json')==br['inputs_sha256']
    for r in bm['inputs']:assert sha(r['snapshot'])==r['sha256']
    active={Path(p).name:Path(p) for p in bm['source_files']}
    old=bind(ROOT/'output/toolchain/上板启动复位取消同源证据_2026-10-08.json')
    assert old['board_inputs_sha256']==sha(board/'inputs.json')
    for gate in old['bindings']:
        p=Path(gate['gate']);result=receipt(p)
        assert result['status']==gate['status'] and sha(p/'result.json')==gate['result_sha256']
        bind(p/'result.json')
        for name,digest in gate['sources'].items():assert sha(active[name])==digest
    classes=bind(ROOT/'tmp/frame-execution/skip-credit-gray-source-only1/sync_head_classes.json')
    assert sha(board/'outflow/ti60f225_oob.map.v')==classes['map_sha256']
    assert classes['counts']==dict(combined_vendor_calibration_status=1,held_abort_request=1,
           legacy_fifo_reset_handshake_counter=12,constant_input_no_toggling_data=8,held_abort_acknowledgement=1)
    for path,digest in classes['source_sha256'].items():
        p=active.get(Path(path).name,board/Path(path).name);assert sha(p)==digest
        evidence[str(p)]=digest
    grayroot=ROOT/'tmp/frame-execution/skip-credit-seed3-gray-release1'
    gray=bind(grayroot/'result.json');gm=bind(grayroot/'inputs.json')
    assert sha(grayroot/'inputs.json')==gray['inputs_sha256']
    assert gray['endpoints']==70 and gray['buses']==10 and gray['max_skew_ns']<gray['limit_ns']
    for p,h in gm['original'].items():assert sha(p)==h
    # The memory vendor boundary is specifically the already tested V4 5.16
    # wrapper, receiver retime and init ROM, not engboard pins. The public AXI
    # bridge was subsequently optimized and has its own exact-source tests.
    vr=ROOT/'tmp/frame-execution/production-ddr-cosim6'
    v=bind(vr/'result.json');vi=bind(vr/'inputs.json')
    assert v['status']=='PRODUCTION_DPI_REAL_DDR_FOUR_MASTER_EQUIVALENCE_PASS' and v['real_calibration']
    assert sha(vr/'inputs.json')==v['inputs_sha256'] and sha(vr/'simulation.log')==v['log_sha256']
    vf={Path(r['path']).name:Path(r['path']) for r in vi['files']}
    for r in vi['files']:assert sha(r['path'])==r['sha256']
    for name in ('ddr516_rx_retime.sv','ddr516_v4_controller.sv'):
        assert sha(board/name)==sha(vf[name]);evidence[str(board/name)]=sha(board/name)
    bridge=ROOT/'tmp/frame-execution/split-io-bridge1'
    for gate_name in ('split-io-bridge1','split-io-mapped1'):
        gate=ROOT/'tmp/frame-execution'/gate_name
        gr=bind(gate/'result.json');gi=bind(gate/'inputs.json')
        assert gr['status']=='DDR516_AXI_BRIDGE_PASS' and len(gr['results'])==4
        assert sha(gate/'inputs.json')==gr['inputs_sha256']
        for row in gi['inputs']:assert sha(row['snapshot'])==row['sha256']
        for row in gr['results']:assert sha(gate/f"seed{row['seed']}.log")==row['log_sha256']
    assert sha(board/'ddr516_axi_address_bridge.sv')==sha(bridge/'ddr516_axi_address_bridge.sv')
    assert sha(board/'ddr3_controller.bin')==vi['controller_rom_sha256']
    # Different protected simulation/synthesis views: compare only public top.
    def public(p):return Path(p).read_text().split('endmodule',1)[0]
    assert public(board/'ddr516_reference.sv')==public(vf['ddr516_reference.sv'])
    wrapper=(board/'ddr516_v4_controller.sv').read_text()
    for s in ('ready_meta<=calibration_done && calibration_pass;ready_sync<=ready_meta;',
              '.reset_n(rstn && ready_sync)', '.cal_ena(1\'b1)'):
        assert s in wrapper
    raw=active['raw_stream_ingress.sv'].read_text()
    for s in ('assign abort_request=phase==WAIT_ACK;', 'WAIT_ACK:if(ack_sync)phase<=WAIT_RELEASE;',
              'WAIT_RELEASE:if(!ack_sync)phase<=RUN;', 'assign ack=request_sync && !capture_busy;'):
        assert s in raw
    fifo=active['guarded_async_fifo.sv'].read_text()
    assert 'RESET_GUARD_CYCLES = 20' in fifo
    assert 's_ready = wr_guard_done && !fifo_full' in fifo
    assert 'fifo_read_en = rd_guard_done && !fifo_empty' in fifo
    top=active['ti60f225_oob_top.v'].read_text()
    assert '.enable(control_enable && cal_done_sync && fixed_boot_run_allowed)' in top
    assert 'assign led[0]=cal_done && !pipeline_error;' in top
    result=dict(status='CDC_RESET_REVIEW_COMPLETE_FOR_BOUNDED_BRINGUP',board=str(board),required_evidence=evidence,
       findings=[
          dict(group='8 constant sync heads',disposition='No toggling source; no CDC-data waiver needed'),
          dict(group='12 legacy FIFO reset counter heads',disposition='Reset-only convergence; both clocks must run before traffic. Original two-domain reset handshake plus 20-cycle guards retained; six ratio/seed tests match exact source'),
          dict(group='abort request / acknowledgement',disposition='Held four-phase handshake; source waits for ACK rise then ACK fall. ACK only after capture outstanding traffic drains. Exact multi-seed cancellation regressions bound'),
          dict(group='calibration done AND pass',disposition='Calibration-only eligibility followed by two destination registers; no workload before host load/CRC/permit. Require stable board calibration and bounded DDR readback before runtime upload'),
          dict(group='70 routed Gray endpoints',disposition='10 nonempty groups; selected-route max 0.057ns < 2.5ns; no constraint relaxed'),
          dict(group='vendor 3 read control / 11 async reset endpoints',disposition='Retained 5.16 vendor clock/reset interface and V4 wrapper; no edits to protected internals. Normal calibration and four-master short test pass; analog recovery/removal behavior remains vendor/board boundary')],
       commissioning_conditions=['Use only selected seed3 immutable bitstream',
          'After fresh configuration wait at least 1 second with stable power/clocks; observe calibration indicator',
          'Inspect idle control then run bounded 4KiB DDR two-pattern CRC test before any upload/start',
          'Any error/timeout stops the host; reconfigure before retry, do not reuse a partial load',
          'Complete payload readback and CRC then allow 1 frame initially; hardware 5-second run stop remains'],
       full_vendor_internal_cdc_proof=False,analog_metastability_proven=False,
       public_bridge_separately_verified=True,formal_board_acceptance=False,exceptions_added=False)
    assert not output.exists();write_json(output,result)
    print(result['status'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();review(a.output)
