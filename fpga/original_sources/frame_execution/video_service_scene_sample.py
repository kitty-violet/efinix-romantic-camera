"""Read a sparse view of an already locked, stopped input frame; no writes.

80x45 point samples of the 1280x720 q8 frame, not a full frame readback.
No USB by default; exact deployment, programming receipt and tests required.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
from run_units import ROOT, sha
from video_service_board import bound, ControlBus
from video_service_input_isp_prepared import prepared_guard
from video_service_observation import fresh, path_status
from windows_process_tree_safe import stop_launched_tree

BASE = 0xe1000000
WIDTH, HEIGHT, STEP = 1280, 720, 16


def addresses():
    result = [BASE + 3 * (y * WIDTH + x)
              for y in range(0, HEIGHT, STEP) for x in range(0, WIDTH, STEP)]
    if len(result) != 3600 or any(a % 4 or not BASE <= a < BASE + WIDTH * HEIGHT * 3 - 3 for a in result):
        raise ValueError('Sparse input sample bounds')
    return result


def rgb(word):
    if type(word) is not int or not 0 <= word <= 0xffffffff:
        raise ValueError('32-bit read word required')
    return bytes((word >> (8 * k) & 255) ^ 128 for k in range(3))


def locked(path):
    if path['mode'] != 1 or not path['inspect_enabled'] or path['inspect_result'] or path['inspect_bank'] not in range(3):
        raise ValueError('Stopped INPUT_PREVIEW input window must already be locked')
    return tuple(path[k] for k in ('mode', 'inspect_enabled', 'inspect_result', 'inspect_bank'))


def same_capture(before, after, first, last):
    if locked(first) != locked(last):
        raise ValueError('Inspection bank changed')
    for key in ('phase', 'observed_coefficient_crc'):
        if before['snapshot'][key] != after['snapshot'][key]:
            raise ValueError('Stopped identity changed')
    if before['snapshot']['apb']['5c'] != after['snapshot']['apb']['5c']:
        raise ValueError('Committed frame changed or reset')


def read_samples(bus, out, save):
    data = bytearray()
    for i, address in enumerate(addresses()):
        data.extend(rgb(bus.read(address)))
        out['pixels_read'] = i + 1
        if (i + 1) % 80 == 0:
            save()
    return bytes(data)


def worker(a, m, receipt):
    out = dict(status='CONNECTING', hardware_access=False, ddr_data_writes=0,
               flash_written=False, cnn_started=False, complete_frame=False,
               source_size=[WIDTH, HEIGHT], sample_size=[80, 45], step=STEP,
               manifest_sha256=sha(a.manifest), program_receipt_sha256=sha(a.program_receipt),
               helper_sha256=sha(__file__), pixels_read=0)
    def save(): a.report.write_text(json.dumps(out, indent=2), encoding='utf-8')
    save(); link = bus = None
    try:
        from fixed_jtag_physical_b import PhysicalBTransport
        from stable_jtag_reply import StableReplyTransport
        link = PhysicalBTransport(receipt['url']); out['hardware_access'] = True
        bus = ControlBus(StableReplyTransport(link), writing=False, read_window=True, seconds=100)
        def path():
            for _ in range(112):
                row = fresh(link, m['service_diagnostic_positions'], m['camera_diagnostic_slots'], [], lambda: None)
                value = path_status(row)
                if value is not None: return value
            raise TimeoutError('No current inspection status')
        out['before'] = prepared_guard(bus); out['path_before'] = path()
        locked(out['path_before']); out['status'] = 'READING_SPARSE_INPUT'; save()
        data = read_samples(bus, out, save)
        out['after'] = prepared_guard(bus); out['path_after'] = path()
        same_capture(out['before'], out['after'], out['path_before'], out['path_after'])
        target = a.report.with_suffix('.ppm')
        with target.open('xb') as f: f.write(b'P6\n80 45\n255\n' + data)
        out.update(status='SPARSE_INPUT_READ_ONLY_PASS', image=str(target), image_sha256=sha(target))
    except Exception as exc:
        out.update(status='FAIL_PREFIX_SAVED', error=repr(exc)); raise
    finally:
        if bus: out['control'] = dict(transactions=bus.transactions, errors=bus.errors, attempts=bus.attempts)
        if link: link.close()
        save()


def main():
    p = argparse.ArgumentParser()
    for name in ('manifest', 'program-receipt', 'report', 'test-receipt'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--execute', action='store_true'); p.add_argument('--supervised', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    for name in ('manifest', 'program_receipt', 'report', 'test_receipt'):
        setattr(a, name, getattr(a, name).resolve())
    m, receipt = bound(a.manifest, a.program_receipt)
    proof = json.loads(a.test_receipt.read_text())
    if proof.get('status') != 'PASS' or proof.get('tests', 0) < 8 or proof.get('helper_sha256') != sha(__file__) or proof.get('test_sha256') != sha(Path(__file__).with_name('test_video_service_scene_sample.py')):
        raise ValueError('Bound focused tests required')
    if not a.execute: print('SCENE_SAMPLE_OFFLINE_BOUND_NO_USB'); return
    if not a.supervised or a.report.exists() or not a.report.is_relative_to(ROOT / 'tmp/board-test'):
        raise ValueError('New supervised report required')
    if a.worker: worker(a, m, receipt); return
    cmd = [sys.executable, __file__, '--manifest', str(a.manifest), '--program-receipt', str(a.program_receipt),
           '--report', str(a.report), '--test-receipt', str(a.test_receipt), '--execute', '--supervised', '--worker']
    start = time.monotonic()
    with a.report.with_suffix('.log').open('x') as log:
        child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try: rc = child.wait(timeout=140); host = dict(exit_code=rc, seconds=time.monotonic()-start)
        except subprocess.TimeoutExpired: rc=124; host=dict(exit_code=rc, cleanup=stop_launched_tree(child))
    a.report.with_suffix('.host.json').write_text(json.dumps(host), encoding='utf-8')
    print(json.dumps(host)); raise SystemExit(rc)


if __name__ == '__main__': main()
