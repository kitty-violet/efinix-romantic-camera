"""Bounded, allowlisted read-only DCO1 measurements; no DDR payload access."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from fixed_jtag_loader import APB, command_scan, decode_response
from read_first_fault import fresh
from windows_process_tree import stop_launched_tree

MAGIC = 0x44434f31
OFFSETS = frozenset((0, 0x6c, *range(0x70, 0x98, 4)))
CLOCKS = ('core', 'tdqss', 'tac', 'twd')


def read_word(transport, offset, rows, save, timeout=2):
    if isinstance(offset, bool) or offset not in OFFSETS:
        raise ValueError('Read address is outside the fixed diagnostic allowlist')
    row = dict(address=f'{APB+offset:08x}', status='REQUESTED')
    rows.append(row)
    save()
    try:
        packet, bits = command_scan(False, APB+offset)
        transport.scan(packet, bits)
        deadline = time.monotonic()+timeout
        for _ in range(200):
            if time.monotonic() >= deadline:
                break
            value = decode_response(transport.scan(1, 36))
            if value is not None:
                row.update(status='READ', value=f'{value:08x}')
                save()
                return value
        raise TimeoutError('Read-only diagnostic request timed out')
    except Exception as exc:
        row.update(status='FAIL', error=repr(exc))
        save()
        raise


def decode_status(value, edges, widths, reset):
    if value >> 23:
        raise ValueError('Reserved DCO1 status bits set')
    return dict(phases_seen=f'{value >> 12 & 255:02x}',
                phase_codes=[i for i in range(8) if value >> (12+i) & 1],
                selections_seen=f'{value >> 7 & 31:02x}',
                shift_ena=bool(value & 64), cke=bool(value & 32),
                ddr_reset_n=bool(value & 16), system_reset_n=bool(value >> 20 & 1),
                age_saturated=bool(value >> 21 & 1), event_saturated=bool(value >> 22 & 1),
                reset_rises=edges & 65535, reset_falls=edges >> 16,
                system_reset_falls=reset >> 16, current_reset_age_ticks=reset & 65535,
                last_low_ticks=widths & 65535, last_high_ticks=widths >> 16,
                tick_us=256/150, note='Words are individually coherent, not an atomic multiword snapshot. Saturated widths are lower bounds.')


def frequency_bounds(first, second):
    # Each count and the timer use divide-by-256, so the factors cancel.
    # Add two ticks for quantization; do not claim exact phase/frequency.
    d = lambda b, a: (b-a) & 0xffffffff
    minimum = d(second['tick_before'], first['tick_after'])-2
    maximum = d(second['tick_after'], first['tick_before'])+2
    if not 0 < minimum <= maximum < 0x80000000:
        raise ValueError('Timer brackets overlap, stall, or have ambiguous wrap')
    return {name: dict(delta=d(second['counts'][name], first['counts'][name]),
                       mhz_lower=max(0, d(second['counts'][name], first['counts'][name])-2)*150/maximum,
                       mhz_upper=(d(second['counts'][name], first['counts'][name])+2)*150/minimum)
            for name in CLOCKS}


def healthy_snapshot(snapshot):
    if not (snapshot['levels']['system_reset_n'] and not snapshot['levels']['control_reset']
            and snapshot['payload'] and snapshot['payload']['armed'] and not snapshot['payload']['captured']):
        raise RuntimeError('Control startup/fault snapshot is not healthy; no further requests')


def worker(url, report):
    from fixed_jtag_physical_b import PhysicalBTransport
    result = dict(status='READING', url=url, hardware_access=True, read_only=True,
                  ddr_payload_accesses=0, flash_writes=0, fpga_programmed=False,
                  reads=[], scans=[], snapshots=[], groups=[], board_pass=False,
                  frequency_reference_mhz=150,
                  frequency_scope='Relative to the process clock configured at 150MHz; not an independent oscillator or jitter measurement.')
    def save():
        report.write_text(json.dumps(result, indent=2), encoding='utf-8')
    transport = None
    save()
    try:
        transport = PhysicalBTransport(url)
        result.update(idcode=f'{transport.idcode:08x}', usb_binding=transport.usb_binding)
        result['snapshots'].append(fresh(transport, result['scans'], save))
        healthy_snapshot(result['snapshots'][-1])
        def read(offset):
            return read_word(transport, offset, result['reads'], save)
        if read(0) != 0x54493630 or read(0x70) != MAGIC:
            raise RuntimeError('Identity/DCO1 mismatch; no further requests')
        for group_id in range(3):
            if group_id:
                time.sleep(.25)
            g = dict(tick_before=read(0x84), counts={})
            result['groups'].append(g)
            for name, offset in zip(CLOCKS, (0x74, 0x78, 0x7c, 0x80)):
                g['counts'][name] = read(offset)
            g['tick_after'] = read(0x84)
            edges, widths, status, reset = [read(offset) for offset in (0x88, 0x8c, 0x90, 0x94)]
            g['observation'] = decode_status(status, edges, widths, reset)
            g['ddr_status_6c'] = f'{read(0x6c):08x}'
            save()
        result['frequency_bounds'] = [frequency_bounds(a, b) for a, b in zip(result['groups'], result['groups'][1:])]
        result['snapshots'].append(fresh(transport, result['scans'], save))
        healthy_snapshot(result['snapshots'][-1])
        result['status'] = 'DDR_OBSERVATION_CAPTURED_NOT_BOARD_PASS'
    except Exception as exc:
        result.update(status='FAIL_PREFIX_SAVED', error=repr(exc))
    finally:
        if transport:
            try:
                transport.close()
            except Exception as exc:
                result.update(status='FAIL_PREFIX_SAVED', close_error=repr(exc))
        save()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    assert not a.report.exists()
    a.report.parent.mkdir(parents=True, exist_ok=True)
    if a.worker:
        raise SystemExit(int(worker(a.url, a.report)['status'] != 'DDR_OBSERVATION_CAPTURED_NOT_BOARD_PASS'))
    start = time.monotonic()
    with a.report.with_suffix('.log').open('x') as log:
        child = subprocess.Popen([sys._base_executable, str(Path(__file__).resolve()), '--worker',
                                  '--url', a.url, '--report', str(a.report.resolve())], stdout=log, stderr=subprocess.STDOUT)
        try:
            code = child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            cleanup = stop_launched_tree(child)
            code = 124
            prefix = json.loads(a.report.read_text()) if a.report.exists() else {}
            prefix.update(status='HOST_TIMEOUT_PREFIX_SAVED', cleanup=cleanup)
            a.report.write_text(json.dumps(prefix, indent=2))
    result = json.loads(a.report.read_text()) if a.report.exists() else dict(status='WORKER_FAILED')
    result.update(exit_code=code, host_seconds=round(time.monotonic()-start, 3))
    a.report.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('scans', 'reads')}, indent=2))
    raise SystemExit(code or int(result['status'] != 'DDR_OBSERVATION_CAPTURED_NOT_BOARD_PASS'))


if __name__ == '__main__':
    main()
