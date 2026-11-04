"""Stage accepted A images in DDR only. No mailbox writes or CNN start.

The supervised entry reprograms the frozen, released SRAM image, proves fresh
DDR scratch reads/writes, then uploads and checks all three DDR images. Resident
bank loading belongs to the later bounded-run command, not to this operation.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
import zlib
from pathlib import Path

import ddr_scratch_bringup as scratch
from fixed_jtag_loader import WINDOWS, command_scan, decode_response, validate_payload
from stable_jtag_reply import StableReplyTransport
from windows_process_tree import stop_launched_tree

ROOT, HERE, BUNDLE = scratch.ROOT, scratch.HERE, scratch.BUNDLE
INTEGRATION = ROOT / 'tmp/frame-execution/accepted-a-integration1'
ORDER = ('coefficients-padded', 'compute', 'dma')
MAX_COMMANDS = 380000
sha, load = scratch.sha, scratch.load


def metadata(data):
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                crc32=f'{zlib.crc32(data):08x}')


def payload_checked():
    data = validate_payload(INTEGRATION)
    for name in ORDER:
        if (BUNDLE / (name + '.bin')).read_bytes() != data[name]:
            raise ValueError('Frozen bundle differs from accepted program: ' + name)
    return data


def validate_evidence():
    prior = scratch.validate_evidence()  # full 4555-file physical/source binding
    record = ROOT / 'tmp/board-test/local-drivers-ddr-scratch1.json'
    d = load(record)
    assert d['status'] == 'PHYSICAL_4KIB_DDR_READ_WRITE_CRC_PASS'
    assert d['transport_errors'] == 0 and d['writes'] == 2052 and d['transactions'] == 4132
    assert all(r['status'] == 'PASS' and r['mismatch_count'] == 0 and
               r['actual_crc32'] == r['expected_crc32'] for r in d['patterns'])
    previous_release = ROOT / 'tmp/board-test/local-drivers-ddr-scratch-release1.json'
    assert d['release_sha256'] == sha(previous_release)
    assert load(previous_release) == prior
    data = payload_checked()
    descriptor = INTEGRATION / 'descriptors/result.json'
    program = Path(load(descriptor)['program_path']) / 'program.json'
    files = [Path(__file__), HERE / 'test_accepted_a_ddr_stage.py', record,
             previous_release, INTEGRATION / 'result.json', descriptor,
             INTEGRATION / 'descriptors/schema.json', program,
             HERE / 'accepted_a_assets.py', ROOT / 'network_ab/model_b_contract.json',
             ROOT / 'tmp/board-test/fresh-session-guard.py',
             ROOT / 'tmp/board-test/probe-official-ftdi-session.py']
    return dict(status='ACCEPTED_A_DDR_STAGE_ONLY_RELEASED',
        manifest_sha256=scratch.MANIFEST_SHA, bitstream_sha256=scratch.BIT_SHA,
        payload={n: dict(metadata(data[n]), host_base=f'{WINDOWS[n][0]:08x}',
                        physical_base=f'{WINDOWS[n][0]-0xdc000000:08x}') for n in ORDER},
        total_bytes=sum(map(len, data.values())),
        residual_records=load(descriptor)['residual_records'],
        residual_fields=dict(main_scaled=False, multiplier=1, post_multiplier=131071),
        resident_banks_load_allowed=False, cnn_start_allowed=False, flash_allowed=False,
        mailbox_writes_allowed=False, total_host_timeout_s=1800, per_reply_timeout_s=2,
        max_stage_commands=MAX_COMMANDS, fresh_scratch_required=True,
        historical_ddr_receipt_sha256=sha(record),
        evidence={**prior['evidence'], **{str(p): sha(p) for p in files}})


class StageBus:
    def __init__(self, transport, seconds=1700):
        self.transport = transport
        self.deadline = time.monotonic() + seconds
        self.transactions = self.errors = self.writes = self.reads = 0
        self.attempted_commands = self.attempted_writes = 0

    def access(self, writing, address, value=0):
        image = any(base <= address < base + size for base, size in WINDOWS.values())
        if address % 4 or (not image and (writing or address not in scratch.READ_STATUS)):
            raise ValueError('DDR image-only policy rejected address')
        packet, bits = command_scan(writing, address, value)
        if time.monotonic() >= self.deadline:
            raise TimeoutError('DDR upload total deadline')
        if self.attempted_commands >= MAX_COMMANDS:
            raise RuntimeError('DDR upload command budget')
        self.attempted_commands += 1
        self.attempted_writes += int(writing)
        try:
            self.transport.scan(packet, bits)  # send exactly once, never replay a write
            end = min(self.deadline, time.monotonic() + 2)
            for _ in range(200):
                if time.monotonic() >= end:
                    break
                value = decode_response(self.transport.scan(1, 36))
                if value is not None:
                    self.transactions += 1
                    self.writes += int(writing)
                    self.reads += int(not writing)
                    return value
            raise TimeoutError(f'No stable reply at {address:08x}')
        except Exception:
            self.errors += 1
            raise

    def read(self, address):
        return self.access(False, address)

    def write(self, address, value):
        return self.access(True, address, value)


def transfer(bus, data, out, save):
    # Require exact images and sizes before the first write; caller already
    # bound their SHA, CRC and decoded residual fields to the accepted package.
    if set(data) != set(ORDER) or any(len(data[n]) != WINDOWS[n][1] for n in ORDER):
        raise ValueError('Incomplete or incorrectly sized payload')
    out['guards'].append(scratch.guard(bus)); save()
    for name in ORDER:
        image = data[name]
        row = dict(name=name, **metadata(image), written_bytes=0, verified_bytes=0,
                   mismatch_count=0, status='WRITING')
        out['images'].append(row); save()
        started = time.monotonic()
        for off in range(0, len(image), 4):
            bus.write(WINDOWS[name][0]+off, int.from_bytes(image[off:off+4], 'little'))
            row['written_bytes'] = off + 4
            if (off+4) % 4096 == 0:
                save()
            if (off+4) % 65536 == 0:
                out['guards'].append(scratch.guard(bus)); save()
        row.update(status='WRITTEN_NOT_VERIFIED', write_seconds=round(time.monotonic()-started, 3))
        out['guards'].append(scratch.guard(bus)); save()
    # Read every word AFTER every region has been written. This also catches
    # cross-region address aliasing that immediate per-file checks can miss.
    for row in out['images']:
        name = row['name']; image = data[name]; actual = bytearray()
        row['status'] = 'READING'; started = time.monotonic(); save()
        for off in range(0, len(image), 4):
            value = bus.read(WINDOWS[name][0]+off)
            actual.extend(value.to_bytes(4, 'little'))
            row['verified_bytes'] = off+4
            if actual[-4:] != image[off:off+4]:
                row.update(status='FAIL', mismatch_count=1,
                    first_mismatch=dict(address=f'{WINDOWS[name][0]+off:08x}',
                        expected_hex=image[off:off+4].hex(), actual_hex=actual[-4:].hex()),
                    checked_prefix_sha256=hashlib.sha256(actual).hexdigest())
                save(); raise IOError('DDR image readback mismatch; no retry')
            if (off+4) % 4096 == 0:
                save()
            if (off+4) % 65536 == 0:
                out['guards'].append(scratch.guard(bus)); save()
        observed = metadata(actual)
        if observed != metadata(image):
            raise IOError('DDR full SHA/CRC differs')
        row.update(status='PASS', readback=observed, read_seconds=round(time.monotonic()-started, 3))
        out['guards'].append(scratch.guard(bus)); save()


def check_snapshot(s):
    if not all((s['ddr']['raw_done'], s['ddr']['raw_pass'], s['levels']['ddr_qualified'])):
        raise RuntimeError('Independent DDR qualification failed')
    if not all(p['matches'] for p in s['cell_probes'].values()):
        raise RuntimeError('Primitive probe mismatch')
    f = s['live']['flags']
    if f['boot_fault'] or f['signals']['fatal_bus_error'] or f['signals']['boot_failed']:
        raise RuntimeError('Controller/boot fault')
    if s['adapter']['captured'] or any(s['adapter']['live'][k] for k in ('read_error', 'write_error')):
        raise RuntimeError('Adapter error')
    if s['payload']['captured'] and any(s['payload']['reason'][k] for k in
            ('illegal_control_state', 'fatal_bus_error', 'boot_failed')):
        raise RuntimeError('Controller first-fault capture')


def sequence(report, release):
    from fixed_jtag_physical_b import PhysicalBTransport
    out = dict(status='STARTING', hardware_access=False, images=[], guards=[], stages=[],
               scans=[], snapshots=[], scratch=dict(patterns=[], guards=[]),
               resident_banks_loaded=False, cnn_started=False, flash_written=False,
               release_sha256=sha(release))
    link = stable = bus = scratch_bus = None
    def save():
        report.write_text(json.dumps(out, indent=2), encoding='utf-8')
    def observe():
        s = scratch.fresh(link, out['scans'], save)
        out['snapshots'].append(s); save(); check_snapshot(s)
        return s
    def run(args, label, seconds):
        # The programmer owns <stem>-program.log for its vendor output.
        # Keep our subprocess console separate from that write-once evidence.
        log = report.with_name(report.stem + '-' + label + '-wrapper.log')
        assert not log.exists()
        row = dict(stage=label, status='RUNNING'); out['stages'].append(row); save()
        with log.open('xb') as f:
            child = subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT)
            try:
                code = child.wait(timeout=seconds)
            except subprocess.TimeoutExpired:
                row['cleanup'] = stop_launched_tree(child); row['status'] = 'HOST_TIMEOUT'; save()
                raise
        row.update(status='PASS' if code == 0 else 'FAIL', exit_code=code); save()
        if code:
            raise RuntimeError('Stage failed: ' + label)
    save()
    try:
        r = load(release)
        if r != validate_evidence():
            raise ValueError('Staging release changed')
        data = payload_checked()
        programmed = report.with_name(report.stem+'-program.json')
        if programmed.exists():
            raise ValueError('Preserve prior programming receipt')
        out['hardware_access'] = True; save()
        run([sys.executable, '-B', str(BUNDLE/'program.py'), '--supervised-program',
             '--report', str(programmed)], 'program', 300)
        p = load(programmed)
        assert p['status'] == 'TEMPORARY_JTAG_COMPLETE_PENDING_READ_ONLY_DIAGNOSTICS'
        assert p['bitstream_sha256'] == scratch.BIT_SHA and p['manifest_sha256'] == scratch.MANIFEST_SHA
        out.update(program_receipt=str(programmed), program_receipt_sha256=sha(programmed))
        receipts = []
        for label, mode in [('control-before', 'control'), ('id', 'id'), ('control-after', 'control')]:
            receipt = report.with_name(report.stem+'-fresh-'+label+'.json')
            assert not receipt.exists()
            run([sys.executable, '-B', str(ROOT/'tmp/board-test/probe-official-ftdi-session.py'),
                 str(receipt), mode], 'fresh-'+label, 22)
            receipts.append(load(receipt))
        url = scratch.module(ROOT/'tmp/board-test/fresh-session-guard.py', 'stage_usb_guard').validated_url(*receipts)
        if url != p['url']:
            raise RuntimeError('USB device changed after programming')
        link = PhysicalBTransport(url)
        out.update(url=url, idcode=f'{link.idcode:08x}', usb_binding=link.usb_binding)
        first = observe()
        if first['payload']['captured']:
            raise RuntimeError('Freshly programmed device already has first-fault capture')
        stable = StableReplyTransport(link); scratch_bus = scratch.ScratchBus(stable)
        out['status'] = 'FRESH_SCRATCH'; save()
        scratch.exercise(scratch_bus, out['scratch'], save)
        observe()
        out['status'] = 'STAGING_DDR'; save()
        bus = StageBus(stable)
        transfer(bus, data, out, save)
        last = observe()
        for key in ('sampled_system_reset_events', 'sampled_ddr_lock_drops'):
            assert first['ddr'][key] == last['ddr'][key], 'Reset/PLL changed during transfer'
        commands = bus.transactions + scratch_bus.transactions
        for key in ('accepted_commands', 'replies', 'received_last'):
            assert (last['live'][key]-first['live'][key]) & 255 == commands & 255, key
        assert (last['live']['received_bits']-first['live']['received_bits']) & 255 == commands*75 & 255
        out['status'] = 'PHYSICAL_ACCEPTED_A_DDR_STAGE_READBACK_PASS'
    except Exception as exc:
        out.update(status='FAIL_PREFIX_SAVED', error=repr(exc))
        if link:
            try:
                observe()
            except Exception as later:
                out['failure_snapshot_error'] = repr(later)
    finally:
        for name, obj in [('stage_counts', bus), ('scratch_counts', scratch_bus)]:
            if obj:
                out[name] = {k: getattr(obj, k) for k in
                    ('transactions', 'errors', 'writes', 'reads', 'attempted_commands', 'attempted_writes')}
        if stable:
            out.update(reply_poll_pairs=stable.poll_pairs, unstable_pairs=stable.unstable_pairs)
        if link:
            try:
                link.close()
            except Exception as exc:
                out.update(status='FAIL_PREFIX_SAVED', close_error=repr(exc))
        save()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--report', required=True, type=Path)
    p.add_argument('--release', type=Path)
    p.add_argument('--supervised-stage', action='store_true')
    p.add_argument('--worker', action='store_true')
    a = p.parse_args()
    assert not a.report.exists(), 'Preserve prior evidence'
    if not a.supervised_stage:
        assert not a.worker
        r = validate_evidence()
        a.report.write_text(json.dumps(r, indent=2), encoding='utf-8')
        print(json.dumps({k:v for k,v in r.items() if k != 'evidence'}, indent=2)); return
    assert a.release
    if a.worker:
        r = sequence(a.report, a.release)
        raise SystemExit(r['status'] != 'PHYSICAL_ACCEPTED_A_DDR_STAGE_READBACK_PASS')
    assert load(a.release) == validate_evidence(), 'Release mismatch before hardware access'
    log = a.report.with_suffix('.log'); assert not log.exists()
    started = time.monotonic()
    with log.open('xb') as f:
        proc = subprocess.Popen([sys.executable, '-B', __file__, '--worker', '--supervised-stage',
            '--release', str(a.release), '--report', str(a.report)], stdout=f, stderr=subprocess.STDOUT)
        try:
            code = proc.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            cleanup = stop_launched_tree(proc); code = 124
            r = load(a.report) if a.report.exists() else {}
            r.update(status='HOST_TIMEOUT_PREFIX_SAVED', cleanup=cleanup)
            a.report.write_text(json.dumps(r, indent=2), encoding='utf-8')
    r = load(a.report) if a.report.exists() else dict(status='WORKER_FAILED')
    r.update(exit_code=code, host_seconds=round(time.monotonic()-started, 3))
    a.report.write_text(json.dumps(r, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in r.items() if k not in ('scans','snapshots')}, indent=2))
    raise SystemExit(code)


if __name__ == '__main__':
    main()
