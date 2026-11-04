"""Offline safety/decoding tests; no USB is opened."""
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from read_bringup_status import APB, OFFSETS, ReadOnlyBus, decode, inspect, worker, main


class FakeTransport:
    idcode = 0x10660a79
    usb_binding = {'test': True}
    def __init__(self, registers):
        self.registers = registers
        self.commands = []
    def scan(self, value, count):
        if count == 77:
            command = value >> 2
            assert (command >> 72) & 1 == 0, 'No write allowed'
            assert (command >> 73) & 3 == 2
            self.address = (command >> 8) & 0xffffffff
            self.commands.append(self.address)
            return 0
        assert count == 36 and value == 1
        return (self.registers[self.address - APB] << 4) | 4


class Tests(unittest.TestCase):
    def test_read_only(self):
        t = FakeTransport({off: off for off in OFFSETS})
        b = ReadOnlyBus(t)
        for off in OFFSETS:
            self.assertEqual(b.read(APB + off), off)
        for address in (APB + 1, APB + 0x10, 0xe1f00000, 0xf9000f00, -1):
            with self.assertRaises(ValueError):
                b.read(address)
        self.assertEqual(len(t.commands), 5)
        self.assertFalse(hasattr(b, 'write'))

    def test_three_samples(self):
        t = FakeTransport({0: 0x54493630, 4: 0x10, 8: 0, 12: 0xb1000001, 0x6c: 0x001800df})
        result = {}
        with patch('read_bringup_status.time.sleep'):
            inspect(t, result, lambda: None)
        self.assertEqual(result['transactions'], 15)
        self.assertEqual(len(result['samples']), 3)
        self.assertTrue(result['samples'][0]['decoded']['pipeline_errors']['owner'])

    def test_identity_reject(self):
        for regs, expected in (({0: 0}, 1), ({0: 0x54493630, 4: 0, 8: 0, 12: 0}, 4)):
            t = FakeTransport(regs)
            result = {}
            with self.assertRaises(ValueError):
                inspect(t, result, lambda: None)
            self.assertEqual(len(t.commands), expected)
            self.assertTrue(result['samples'][0]['registers'])

    def test_decode_all_bits(self):
        d = decode(0xb10fffff, 0xffffffff)
        self.assertTrue(all(d['pipeline_errors'].values()))
        self.assertEqual((d['datapath_fault'], d['program_fault']), (255, 15))
        self.assertTrue(d['arbiter_error'] and d['cpu_error'])
        self.assertEqual((d['calibration_fail_log'], d['calibration_shift_value']), (255, 7))
        self.assertTrue(all(d['clock_reset_calibration'].values()))
        self.assertTrue(all(d['sampled_handshakes'].values()))
        for k, name in enumerate(d['clock_reset_calibration']):
            row = decode(0xb1000000, 1 << k)['clock_reset_calibration']
            self.assertEqual([n for n, v in row.items() if v], [name])
        for k, name in enumerate(d['sampled_handshakes']):
            row = decode(0xb1000000, 1 << (21 + k))['sampled_handshakes']
            self.assertEqual([n for n, v in row.items() if v], [name])
        with self.assertRaises(ValueError):
            decode(0xb1f00000, 0)

    def test_close_failure_exits_nonzero(self):
        t = FakeTransport({0: 0x54493630, 4: 0, 8: 0, 12: 0xb1000000, 0x6c: 0})
        def failed_close():
            raise OSError('injected close failure')
        t.close = failed_close
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / 'result.json'
            with patch('fixed_jtag_physical_b.PhysicalBTransport', return_value=t), \
                 patch('read_bringup_status.time.sleep'), \
                 patch('sys.argv', ['read', '--worker', '--url', 'fake', '--report', str(report)]):
                with self.assertRaises(SystemExit) as caught:
                    main()
            self.assertEqual(caught.exception.code, 1)
            result = json.loads(report.read_text())
            self.assertEqual(result['status'], 'FAIL')
            self.assertEqual(result['transactions'], 15)
            self.assertIn('injected close failure', result['close_error'])

    def test_failure_preserves_prefix_and_closes(self):
        t = FakeTransport({0: 0x54493630})
        with tempfile.TemporaryDirectory() as tmp, \
             patch('fixed_jtag_physical_b.PhysicalBTransport', return_value=t), \
             patch.object(t, 'close', create=True) as closed:
            report = Path(tmp) / 'result.json'
            with self.assertRaises(KeyError):
                worker('fake', report)
            result = json.loads(report.read_text())
            self.assertEqual(result['status'], 'FAIL')
            self.assertEqual(result['transactions'], 1)
            self.assertEqual(result['samples'][0]['registers'], {'00': 0x54493630})
            closed.assert_called_once()

    def test_outer_rejects_zero_exit_with_failed_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / 'result.json'
            def finish(timeout):
                self.assertEqual(timeout, 20)
                report.write_text(json.dumps({'status': 'FAIL', 'close_error': 'injected'}))
                return 0
            with patch('read_bringup_status.subprocess.Popen') as launched, \
                 patch('sys.argv', ['read', '--url', 'fake', '--report', str(report)]), \
                 patch('builtins.print'):
                launched.return_value.wait.side_effect = finish
                with self.assertRaises(SystemExit) as caught:
                    main()
            self.assertEqual(caught.exception.code, 1)
            self.assertEqual(json.loads(report.read_text())['exit_code'], 1)


if __name__ == '__main__':
    unittest.main()
