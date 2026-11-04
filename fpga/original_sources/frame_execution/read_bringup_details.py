"""Bounded, read-only USER1 diagnostics including idle loader registers. Never load, start, or access DDR.

Use only after independently gated diagnostic configuration. Each sample is
non-atomic; persistent values across samples are evidence, handshake bits are
not transaction counters. An outer process enforces a 20-second host limit.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from fixed_jtag_loader import APB, command_scan, decode_response
from windows_process_tree import stop_launched_tree

OFFSETS = (0, 4, 8, 12, 0x6c, 0x64, 0x10, 0x14, 0x18)


def decode(pipeline, ddr):
    if pipeline >> 24 != 0xb1 or pipeline & 0x00f00000:
        raise ValueError('Diagnostic signature mismatch; no further operation')
    errors = ('owner', 'capture', 'production', 'reader', 'fabric', 'capture_merge')
    pins = ('system_reset_n', 'key1_n', 'mipi_pll_locked', 'ddr_pll_locked',
            'user_pll_locked', 'calibration_qualified', 'calibration_done', 'calibration_pass')
    handshakes = ('awvalid', 'awready', 'wvalid', 'wready', 'bvalid', 'bready',
                  'arvalid', 'arready', 'rvalid', 'rready', 'boot_run_allowed')
    return dict(pipeline_errors={n: bool(pipeline & (1 << k)) for k, n in enumerate(errors)},
                datapath_fault=(pipeline >> 6) & 255, program_fault=(pipeline >> 14) & 15,
                arbiter_error=bool(pipeline & (1 << 18)), cpu_error=bool(pipeline & (1 << 19)),
                clock_reset_calibration={n: bool(ddr & (1 << k)) for k, n in enumerate(pins)},
                calibration_fail_log=(ddr >> 8) & 255, calibration_shift_value=(ddr >> 16) & 7,
                ddr_reset_n=bool(ddr & (1 << 19)), ddr_cke=bool(ddr & (1 << 20)),
                sampled_handshakes={n: bool(ddr & (1 << (21 + k))) for k, n in enumerate(handshakes)})


class ReadOnlyBus:
    def __init__(self, transport):
        self.transport = transport
        self.transactions = 0

    def read(self, address):
        if address not in {APB + off for off in OFFSETS}:
            raise ValueError('Read-only diagnostic address policy')
        value, bits = command_scan(False, address)
        self.transport.scan(value, bits)
        deadline = time.monotonic() + 2
        for _ in range(200):
            if time.monotonic() >= deadline:
                break
            response = decode_response(self.transport.scan(1, 36))
            if response is not None:
                self.transactions += 1
                return response
        raise TimeoutError(f'Diagnostic APB read timeout {address:08x}')


def inspect(transport, result, save):
    bus = ReadOnlyBus(transport)
    result['idcode'] = f'{transport.idcode:08x}'
    result['usb_binding'] = transport.usb_binding
    result['samples'] = []
    try:
        for _ in range(3):
            registers = {}
            # Persist each successful read so any failure preserves its prefix.
            sample = dict(registers=registers)
            result['samples'].append(sample)
            for off in OFFSETS:
                registers[f'{off:02x}'] = bus.read(APB + off)
                result['transactions'] = bus.transactions
                save()
                if off == 0 and registers['00'] != 0x54493630:
                    raise ValueError('APB identity mismatch')
                if off == 12 and registers['0c'] >> 24 != 0xb1:
                    raise ValueError('Not the diagnostic hardware revision')
            sample['decoded'] = decode(registers['0c'], registers['6c'])
            sample['loader_detail'] = dict(error_code=(registers['64'] >> 16) & 15,
                image_tag=registers['64'] & 65535, begin_version=registers['10'],
                begin_image_bytes=registers['14'], begin_tag=registers['18'])
            save()
            time.sleep(.05)
    finally:
        result['transactions'] = bus.transactions


def worker(url, report):
    from fixed_jtag_physical_b import PhysicalBTransport
    result = dict(status='READING', url=url, read_only=True, ddr_accesses=0,
                  fpga_programmed=False, transactions=0)
    def save():
        report.write_text(json.dumps(result, indent=2), encoding='utf-8')
    save()
    transport = None
    try:
        transport = PhysicalBTransport(url)
        inspect(transport, result, save)
        result['status'] = 'DIAGNOSTIC_READ_COMPLETE_NOT_BOARD_PASS'
    except Exception as exc:
        result.update(status='FAIL', error=repr(exc))
        raise
    finally:
        if transport:
            try:
                transport.close()
            except Exception as exc:
                result.update(status='FAIL', close_error=repr(exc))
        save()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', required=True)
    p.add_argument('--report', required=True, type=Path)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    assert not a.report.exists(), 'Preserve prior evidence'
    a.report.parent.mkdir(parents=True, exist_ok=True)
    if a.worker:
        result = worker(a.url, a.report)
        if result['status'] != 'DIAGNOSTIC_READ_COMPLETE_NOT_BOARD_PASS':
            raise SystemExit(1)
        return
    start = time.monotonic()
    log = a.report.with_suffix('.log')
    assert not log.exists()
    with log.open('w') as output:
        proc = subprocess.Popen([sys._base_executable, str(Path(__file__).resolve()),
                                 '--worker', '--url', a.url, '--report', str(a.report.resolve())],
                                stdout=output, stderr=subprocess.STDOUT)
        try:
            code = proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            cleanup = stop_launched_tree(proc)
            prefix = json.loads(a.report.read_text()) if a.report.exists() else {}
            prefix.update(status='HOST_TIMEOUT', cleanup=cleanup)
            a.report.write_text(json.dumps(prefix, indent=2), encoding='utf-8')
            code = 124
    result = json.loads(a.report.read_text()) if a.report.exists() else dict(status='WORKER_FAILED')
    if not code and result.get('status') != 'DIAGNOSTIC_READ_COMPLETE_NOT_BOARD_PASS':
        code = 1
    result.update(exit_code=code, host_seconds=round(time.monotonic() - start, 3))
    a.report.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    if code:
        raise SystemExit(code)


if __name__ == '__main__':
    main()
