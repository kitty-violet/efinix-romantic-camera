"""Supervised, bounded, read-only timing of live commits and observation words.

No control writes or CNN commands. Each word keeps its own host interval;
asynchronously assembled pages are never subtracted as a simultaneous snapshot.
"""
import argparse, json, math, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
from run_units import ROOT, sha
from video_service_board import bound, ControlBus
from video_service_observation import fresh
from video_service_frame_accounting import delta
from read_video_service_pages import PageAssembler
from fixed_jtag_loader import APB, MB
from windows_process_tree_safe import stop_launched_tree


WORDS = {
    'process': {11: 'captured', 10: 'skipped', 9: 'orphan_pixels',
                8: 'compute_admitted', 7: 'committed', 6: 'repeated_reads',
                5: 'dropped_results'},
    'health': {12: 'source_area_frames'},
    'display': {13: 'physical_frames', 12: 'complete_frames',
                11: 'distinct_frames', 10: 'repeated_frames'},
}


def rate(rows, name, maximum_events=8192):
    if len(rows) < 2:
        raise ValueError('At least two observations are required')
    for r in rows:
        a, b = r['before_seconds'], r['after_seconds']
        if not math.isfinite(a) or not math.isfinite(b) or a > b:
            raise ValueError('Invalid sample interval')
    for a, b in zip(rows, rows[1:]):
        if b['before_seconds'] < a['after_seconds']:
            raise ValueError('Overlapping/reversed observations')
        delta(name, a['count'], b['count'], maximum_events)
    first, last = rows[0], rows[-1]
    count = delta(name, first['count'], last['count'], maximum_events)
    low = last['before_seconds'] - first['after_seconds']
    high = last['after_seconds'] - first['before_seconds']
    if low <= 0:
        raise ValueError('No positive measurement interval')
    return dict(events=count, interval_seconds=[low, high],
                rate_bounds_hz=[count/high, count/low],
                mean_event_interval_ms_bounds=[low*1000/count, high*1000/count] if count else None,
                samples=len(rows))


def summarize(report):
    epochs = [r['chunk_words'][0] >> 16 for r in report['chunks']
              if r.get('page_kind') == 'health' and r.get('chunk_index') == 13]
    if len(epochs) < 2 or len(set(epochs)) != 1 or epochs[0] == 65535:
        raise ValueError('Reset history missing, changed or saturated')
    modes = [r['chunk_words'][0] & 3 for r in report['chunks']
             if r.get('page_kind') == 'path' and r.get('chunk_index') == 14]
    if len(modes) < 2 or any(modes):
        raise ValueError('LIVE_A mode not consistently observed')
    if any(r['phase'] != 4 or r['error'] for r in report['samples']):
        raise ValueError('Run stopped or control error occurred')
    out = dict(ai_commits=rate(report['samples'], 'committed'), reset_epoch=epochs[0],
               word_rates={}, unavailable={}, loss_fraction=None)
    for kind, indices in WORDS.items():
        for index, name in indices.items():
            rows = [dict(count=r['chunk_words'][0], before_seconds=r['before_seconds'],
                         after_seconds=r['after_seconds']) for r in report['chunks']
                    if r.get('page_kind') == kind and r.get('chunk_index') == index
                    and r.get('measurement_valid')]
            try:
                item = rate(rows, name)
                item.update(source_page=kind, chunk_index=index,
                            host_intervals_bound_value_capture=kind != 'display')
                if kind == 'display':
                    item['caveat'] = 'Pixel mailbox cache age is not independently timestamped; rate is approximate.'
                out['word_rates'][name] = item
            except ValueError as exc:
                out['unavailable'][name] = str(exc)
    out['notes'] = [
        'APB completed AI result commits exclude repeated HDMI refreshes.',
        'Each counter has its own time interval. Do not subtract rates to claim exact frame loss.',
        'source_area_frames counts downscaled SOF handshakes, not all camera frames.',
        'Use separately stopped/drained boundaries to calculate admitted-frame failure counts.',
    ]
    return out


def worker(a):
    m, receipt = bound(a.manifest, a.program_receipt)
    if m['diagnostic_schema'] not in (5,6):
        raise ValueError('This reader requires the current 32-bit observation schema')
    report = dict(status='READING', started_utc=datetime.now(timezone.utc).isoformat(),
                  manifest_sha256=sha(a.manifest), bitstream_sha256=m['bitstream_sha256'],
                  reader_sha256=sha(__file__), requested_seconds=a.seconds, writes=0,
                  samples=[], chunks=[], pages=[], hardware_access=False)
    def save():
        a.report.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    save(); link = bus = None
    try:
        from fixed_jtag_physical_b import PhysicalBTransport
        from stable_jtag_reply import StableReplyTransport
        link = PhysicalBTransport(receipt['url']); report['hardware_access'] = True
        bus = ControlBus(StableReplyTransport(link), writing=False, seconds=a.seconds+20)
        assembler = PageAssembler(m['camera_diagnostic_slots'])
        begin = time.perf_counter(); next_count = begin
        def count_sample():
            phase = bus.read(MB+4); error = bus.read(APB+0x60)
            start = time.perf_counter(); count = bus.read(APB+0x5c); end = time.perf_counter()
            report['samples'].append(dict(phase=phase, error=error, count=count,
                before_seconds=start-begin, after_seconds=end-begin))
            if phase != 4 or error:
                raise RuntimeError('Live run or control state changed')
        count_sample()
        next_count += 2
        while time.perf_counter()-begin < a.seconds:
            if len(report['chunks']) >= 12000:
                raise TimeoutError('Read-only scan budget exceeded')
            start = time.perf_counter()
            row = fresh(link, m['service_diagnostic_positions'], m['camera_diagnostic_slots'], [], lambda: None)
            end = time.perf_counter()
            row.update(before_seconds=start-begin, after_seconds=end-begin)
            report['chunks'].append(row)
            page = assembler.add(row)
            if page is not None:
                page.update(last_chunk_seconds=end-begin)
                report['pages'].append(page)
            if end >= next_count:
                count_sample(); next_count = end+2; save()
        count_sample()
        report.update(status='READ_ONLY_TIMED_LIVE_MEASURED', measurement=summarize(report))
    except Exception as exc:
        report.update(status='FAIL_PREFIX_SAVED', error=repr(exc)); raise
    finally:
        if bus is not None:
            report['control'] = dict(transactions=bus.transactions, errors=bus.errors)
        if link is not None:
            link.close()
        save()
    print(json.dumps(report['measurement']))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--program-receipt', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--seconds', type=int, default=30)
    p.add_argument('--execute', action='store_true'); p.add_argument('--supervised', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    a.manifest=a.manifest.resolve(); a.program_receipt=a.program_receipt.resolve(); a.report=a.report.resolve()
    if not 10 <= a.seconds <= 60:
        raise ValueError('Measurement must be 10..60 seconds')
    if not a.execute:
        bound(a.manifest, a.program_receipt)
        print('TIMED_READER_OFFLINE_BOUND_NO_USB'); return
    if not a.supervised or not a.report.is_relative_to(ROOT/'tmp/board-test') or a.report.exists():
        raise ValueError('Explicit supervision and a new board-test receipt are required')
    if a.worker:
        worker(a); return
    bound(a.manifest, a.program_receipt)
    cmd=[sys.executable, '-u', __file__, '--manifest', str(a.manifest),
         '--program-receipt', str(a.program_receipt), '--report', str(a.report),
         '--seconds', str(a.seconds), '--execute', '--supervised', '--worker']
    with a.report.with_suffix('.log').open('x') as log:
        child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        start=time.monotonic()
        try:
            rc=child.wait(timeout=a.seconds+60); result=dict(exit_code=rc,seconds=time.monotonic()-start)
        except subprocess.TimeoutExpired:
            rc=124; result=dict(exit_code=rc,seconds=time.monotonic()-start,cleanup=stop_launched_tree(child))
    a.report.with_suffix('.host.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result)); raise SystemExit(rc)


if __name__ == '__main__':
    main()
