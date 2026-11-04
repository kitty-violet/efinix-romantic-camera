"""Full-size public preprocessing regression beside the real-DDR board run.

Uses only RTL frozen for that board. It does not replace CSI/DDR/CNN/HDMI in
the whole-board measurement or borrow this shorter test's time for it.
"""
import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
from run_units import HERE, ROOT, MS, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--board', type=Path, required=True)
    args = parser.parse_args()
    root, board = args.root.resolve(), args.board.resolve()
    assert root.is_relative_to(ROOT / 'tmp')
    root.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((board / 'inputs.json').read_text())
    sources = {Path(row['snapshot']).name: row for row in manifest['inputs']}
    names = ['DC_FIFO.v', 'true_dual_port_ram.v', 'rgb_gain_v1.v',
        'rgb_gain_mul_u8x16_logic.sv', 'line_buffer.v', 'raw_to_rgb.v',
        'debayer_top_2to1.v', 'raw_line_bank_ram.sv', 'raw_line_feeder.sv',
        'guarded_async_fifo.sv', 'rgb_pair_cdc_serializer.sv',
        'legacy_debayer_stream_wrapper.sv', 'rgb_area_downscaler_3to2.sv']
    records = []
    for name in names:
        record = sources[name]
        source = Path(record['snapshot'])
        assert sha(source) == record['sha256'], name
        target = root / name
        shutil.copy2(source, target)
        records.append(dict(source=str(source), snapshot=str(target), sha256=sha(target)))
    original = HERE.parent / 'video_scaler/tb_legacy_debayer_scaler_full_frame.sv'
    body = original.read_text()
    for before, after in [
        ('always #6.734 pair_clk', 'always #6.734 pair_clk'),
        ('forever #3.3335 pixel_clk', 'forever #3.333 pixel_clk'),
        ('.LINE_GAP_CYCLES(3)', '.LINE_GAP_CYCLES(4)'),
        ('.r_gain(16\'d256), .g_gain(16\'d256), .b_gain(16\'d256)',
         '.r_gain(16\'d255), .g_gain(16\'d255), .b_gain(16\'d255)'),
        ('rgb_bilinear_scaler_prototype #(', 'rgb_area_downscaler_3to2 #('),
        ('.UPSCALE(0), .SRC_WIDTH(SRC_WIDTH)', '.SRC_WIDTH(SRC_WIDTH)'),
        ('.m_valid(down_m_valid), .m_ready(1\'b1)', '.m_valid(down_m_valid), .m_ready(sink_ready)'),
        ('if (down_m_valid) begin', 'if (down_m_valid && sink_ready) begin'),
    ]:
        assert body.count(before) == 1, before
        body = body.replace(before, after)
    # The old test drove a continuous ideal pair stream and never stalled the
    # scaler output. Exercise the actual line-reservation boundary instead.
    before = "wire input_valid = rst_n && !input_done;"
    assert before in body
    body = body.replace(before, "integer source_ticks=0;\n    always @(posedge pair_clk) source_ticks<=source_ticks+1;\n    wire input_valid = rst_n && !input_done && source_ticks%23!=0;")
    body = body.replace('integer cycle_count = 0;',
        'integer cycle_count = 0;\n    wire sink_ready = (cycle_count%50000)>=4000;')
    body = body.replace('PASS full legacy Debayer to scaler:', 'PASS BOARD_FROZEN_PREPROCESS_BACKPRESSURE:')
    tb = root / original.name
    tb.write_text(body)
    records.append(dict(source=str(original), snapshot=str(tb), sha256=sha(tb),
        source_sha256=sha(original), stimulus_adapted=True))
    inputs = dict(inputs=records, board_inputs_sha256=sha(board / 'inputs.json'),
        runner_sha256=sha(Path(__file__)), input_geometry=[1920, 1080], output_geometry=[1280, 720],
        source_pause='one cycle in 23', sink_pause='4000 cycles in 50000',
        scope='RAW pairs through real line feeder/Debayer/CDC/area scaler; no CSI/DDR/CNN/HDMI')
    (root / 'inputs.json').write_text(json.dumps(inputs, indent=2))
    def run(command, log_name, timeout):
        start = time.monotonic()
        with (root / log_name).open('w') as log:
            proc = subprocess.run([str(x) for x in command], cwd=root, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=timeout)
        text = (root / log_name).read_text(errors='replace')
        assert proc.returncode == 0 and not re.search(r'(?m)^#?\s*\*\* (Error|Fatal)', text), text[-5000:]
        return text, time.monotonic() - start
    try:
        run([MS / 'vlib.exe', 'work'], 'vlib.log', 30)
        run([MS / 'vlog.exe', '-sv', '-timescale', '1ns/1ps',
             *[row['snapshot'] for row in records]], 'compile.log', 60)
        log, seconds = run([MS / 'vsim.exe', '-c', 'work.tb_legacy_debayer_scaler_full_frame',
            '-do', 'onerror {quit -code 1}; run -all; quit -code 0'], 'simulation.log', 600)
        passes = re.findall(r'^# (PASS BOARD_FROZEN_PREPROCESS_BACKPRESSURE:.*)$', log, re.M)
        assert len(passes) == 1 and not re.search(r'(?m)^# (?:ERROR|FAIL)\b', log)
        for row in records:
            assert sha(row['snapshot']) == row['sha256']
        result = dict(status='BOARD_PREPROCESS_FULL_GEOMETRY_BACKPRESSURE_PASS',
            result=passes[0], seconds=seconds, inputs_sha256=sha(root / 'inputs.json'),
            log_sha256=sha(root / 'simulation.log'), full_board=False,
            note='Framing, counts, known data and backpressure; this run has no CNN or physical DDR.')
        (root / 'result.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    except Exception as exc:
        (root / 'failure.json').write_text(json.dumps(dict(error=repr(exc)), indent=2))
        raise


if __name__ == '__main__':
    main()
