"""Bind existing boot/reset/abort regressions to the current active board RTL.

This closes stale 'not checked on this board' items without rerunning unchanged
tests. It does not certify analog CDC behavior or a physical cable transfer.
"""
import argparse
import json
from pathlib import Path
from run_units import ROOT, sha


def receipt(gate):
    result = json.loads((gate / 'result.json').read_text())
    meta = json.loads((gate / 'inputs.json').read_text())
    assert sha(gate / 'inputs.json') == result['inputs_sha256']
    if 'inputs' in meta:
        for row in meta['inputs']:
            assert sha(row['snapshot']) == row['sha256']
    else:
        assert result['status'] == 'RESET_TAIL_CYCLE_EQUIVALENCE_PASS'
        assert sha(meta['base_source']) == meta['base_sha256']
        assert sha(gate / 'cnn_video_pipeline.sv') == meta['candidate_sha256']
        assert sha(gate / 'reset_views.sv') == meta['reset_views_sha256']
        assert sha(gate / 'tb_reset_tail.sv') == meta['testbench_sha256']
    for name, digest in meta.get('generated', {}).items():
        assert sha(gate / name) == digest
    if 'log_sha256' in result:
        assert sha(gate / 'simulation.log') == result['log_sha256']
    for name, digest in result.get('logs', {}).items():
        assert sha(gate / name) == digest
    for index, row in enumerate(result.get('results', [])):
        assert sha(gate / f'simulation{index}.log') == row['log_sha256']
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--board', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    board, out = a.board.resolve(), a.output.resolve()
    assert board.is_relative_to(ROOT / 'tmp') and out.is_relative_to(ROOT / 'output') and not out.exists()
    manifest = json.loads((board / 'inputs.json').read_text())
    assert sha(board / 'inputs.json') == json.loads((board / 'result.json').read_text())['inputs_sha256']
    for row in manifest['inputs']:
        assert sha(row['snapshot']) == row['sha256']
    active = {Path(f).name: Path(f) for f in manifest['source_files']}
    bindings = []
    cases = [
        ('fixed-soc-jtag-full2', 'PASS_FIXED_SOC_COMPLETE',
         ['resident_weight_bank64.sv', 'coefficient_bank_store_fast.sv', 'cnn_apb_control.sv', 'sapphire_axi32_to_axi128.sv']),
        ('fixed-soc-faults2', 'FIXED_CONTROL_FAULTS_PASS', []),
        ('board-fifo-reset-ratios1', 'LOCAL_COMPACT_VIDEO_CDC_PASS', ['DC_FIFO.v', 'guarded_async_fifo.sv', 'framed_rgb_async_fifo.sv']),
        ('raw-payload-ingress1', 'RAW_STREAM_INGRESS_PASS', ['raw_stream_ingress.sv', 'raw32_validated_packer.sv', 'guarded_async_fifo.sv', 'DC_FIFO.v']),
        ('raw-payload-front1', 'RAW_STREAM_FRONTEND_PASS', ['raw_stream_ingress.sv', 'raw_stream_frontend.sv']),
        ('reset-tail-equivalence1', 'RESET_TAIL_CYCLE_EQUIVALENCE_PASS', []),
    ]
    for name, status, files in cases:
        gate = ROOT / 'tmp/frame-execution' / name
        r = receipt(gate)
        assert r['status'] == status
        hashes = {}
        for file in files:
            assert sha(active[file]) == sha(gate / file), (name, file)
            hashes[file] = sha(active[file])
        if name.startswith('fixed-soc'):
            pieces = ('official_jtag.sv', 'cnn_boot_sequencer.sv', 'cnn_fixed_control_soc.sv')
            assert active['EfxSapphireSoc.v'].read_text() == '\n'.join((gate / file).read_text() for file in pieces)
            hashes['EfxSapphireSoc.v'] = sha(active['EfxSapphireSoc.v'])
        if name == 'reset-tail-equivalence1':
            assert sha(active['cnn_video_pipeline.sv']) == r['candidate_sha256']
            hashes['cnn_video_pipeline.sv'] = r['candidate_sha256']
        bindings.append(dict(gate=str(gate), status=status, result_sha256=sha(gate / 'result.json'), sources=hashes))
    top = active['ti60f225_oob_top.v'].read_text()
    assert '.enable(control_enable && cal_done_sync && fixed_boot_run_allowed)' in top
    soc = active['EfxSapphireSoc.v'].read_text()
    for text in ('RUN_TIMEOUT_CYCLES=750000000', 'BUS_TIMEOUT_CYCLES=4096',
                 'rsp_data<1 || rsp_data>16', 'boot_run_allowed=permit && !fatal_bus_error && !reset',
                 "wire ddr_hit=address[31:24]==8'he1;"):
        assert text in soc, text
    result = dict(status='CURRENT_BOARD_BOOT_RESET_ABORT_EVIDENCE_BOUND',
                  board=str(board), board_inputs_sha256=sha(board / 'inputs.json'), bindings=bindings,
                  boot=dict(coeff_words_verified=2560, independent_run_timeout_cycles=750000000,
                            bus_timeout_cycles=4096, accepted_frame_budget=[1,16],
                            current_soc_exactly_matches_test=True, top_run_permit_connected=True,
                            physical_usb_transport_verified=False, full_descriptor_upload_verified=False),
                  reset_abort=dict(fifo_clock_cases=6, reset_equivalence_cases=4, raw_abort_seeds=3,
                                   analog_metastability_proven=False),
                  remaining=['Physical USB/JTAG host transport and full current descriptor upload',
                             'Final vendor reset/calibration-control CDC contract review',
                             'Stage-specific board connection and programming release'],
                  board_release=False)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','boot','remaining')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
