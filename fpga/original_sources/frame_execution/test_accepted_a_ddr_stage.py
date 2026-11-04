"""Offline checks of the real-sized A staging path and fail-stop policy."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import accepted_a_ddr_stage as h
from test_ddr_scratch_bringup import Link


class AliasLink(Link):
    def scan(self, value, bits):
        if bits == 77:
            word = value >> 2
            address = (word >> 8) & 0xffffffff
            if 0xe1000000 <= address < 0xe1300000:
                address &= ~0x00300000  # lose region address bits on reads AND writes
                word = (word & ~(0xffffffff << 8)) | (address << 8)
                value = word << 2
        return super().scan(value, bits)


class CorruptLink(Link):
    def scan(self, value, bits):
        result = super().scan(value, bits)
        if bits == 77:
            word = value >> 2
            if not ((word >> 72) & 1) and ((word >> 8) & 0xffffffff) == 0xe1200004:
                self.reply ^= 1
        return result


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = h.payload_checked()

    def transfer(self, link):
        self.out = dict(images=[], guards=[])
        bus = h.StageBus(h.StableReplyTransport(link))
        h.transfer(bus, self.data, self.out, lambda: None)
        return bus

    def test_real_payload_entire_roundtrip_and_order(self):
        link = Link(); bus = self.transfer(link)
        self.assertEqual(bus.writes, 187504)
        self.assertEqual(sum(r['verified_bytes'] for r in self.out['images']), 750016)
        self.assertEqual([r['status'] for r in self.out['images']], ['PASS']*3)
        self.assertEqual([r['readback']['crc32'] for r in self.out['images']],
                         ['7c6149bf', 'c89fa72e', 'fc4cc6a5'])
        writes = [i for i, (w,a,v) in enumerate(link.calls) if w]
        reads = [i for i, (w,a,v) in enumerate(link.calls)
                 if not w and any(b <= a < b+s for b,s in h.WINDOWS.values())]
        self.assertGreater(min(reads), max(writes))
        self.assertLess(bus.attempted_commands, h.MAX_COMMANDS)

    def test_cnn_mailbox_apb_scratch_and_boundaries_forbidden(self):
        link = Link(); bus = h.StageBus(link)
        bad = [h.scratch.MB+x for x in range(0, 40, 4)]
        bad += [h.scratch.APB, h.scratch.APB+8, h.scratch.SCRATCH, 0, 0xe0000000]
        for base, size in h.WINDOWS.values():
            bad += [base-4, base+size, base+1, base+size-1]
        for address in bad:
            with self.assertRaises(ValueError):
                bus.write(address, 0x434e4e31)
        self.assertEqual(link.calls, [])

    def test_error_and_timeout_never_repeat_write(self):
        for fault in ('error', 'timeout'):
            link = Link(fault); bus = h.StageBus(h.StableReplyTransport(link))
            with self.assertRaises((IOError, TimeoutError)):
                bus.write(0xe1200000, 123)
            self.assertEqual(len(link.calls), 1)
            self.assertEqual(bus.attempted_writes, 1)
            self.assertEqual(bus.errors, 1)

    def test_deadline_budget_invalid_data_no_scan(self):
        link = Link(); bus = h.StageBus(link); bus.deadline = 0
        with self.assertRaises(TimeoutError): bus.write(0xe1200000, 0)
        bus = h.StageBus(link); bus.attempted_commands = h.MAX_COMMANDS
        with self.assertRaises(RuntimeError): bus.write(0xe1200000, 0)
        bus = h.StageBus(link)
        for value in (-1, 2**32):
            with self.assertRaises(ValueError): bus.write(0xe1200000, value)
        self.assertEqual(link.calls, [])

    def test_bad_calibration_or_nonidle_no_writes(self):
        for a,v in ((h.scratch.APB+0x6c, 0), (h.scratch.APB+4, 1),
                    (h.scratch.APB+8, 3), (h.scratch.MB+4, 2)):
            link = Link(); link.mem[a] = v
            with self.assertRaises(RuntimeError): self.transfer(link)
            self.assertFalse(any(w for w,a,v in link.calls))

    def test_corruption_records_exact_first_word_and_stops(self):
        link = CorruptLink()
        with self.assertRaises(IOError): self.transfer(link)
        row = self.out['images'][0]
        self.assertEqual(row['first_mismatch']['address'], 'e1200004')
        self.assertEqual(row['verified_bytes'], 8)
        self.assertEqual(row['mismatch_count'], 1)
        self.assertEqual(link.calls[-1][:2], (0, 0xe1200004))

    def test_cross_region_alias_caught_after_all_writes(self):
        with self.assertRaises(IOError): self.transfer(AliasLink())
        self.assertEqual(self.out['images'][0]['status'], 'FAIL')
        self.assertTrue(all(r['written_bytes'] == r['bytes'] for r in self.out['images']))

    def test_partial_wrong_sized_payload_rejected_before_usb(self):
        for data in ({}, dict(self.data, compute=self.data['compute'][:-4])):
            link = Link()
            with self.assertRaises(ValueError):
                h.transfer(h.StageBus(link), data, dict(images=[], guards=[]), lambda:None)
            self.assertEqual(link.calls, [])

    def test_accepted_payload_tamper_rejected(self):
        bad = dict(self.data)
        raw = bytearray(bad['coefficients-padded']); raw[-1] = 1
        bad['coefficients-padded'] = raw
        with patch.object(h, 'validate_payload', return_value=bad):
            with self.assertRaisesRegex(ValueError, 'differs'):
                h.payload_checked()

    def test_only_early_stall_snapshot_is_permitted(self):
        report = h.load(h.ROOT/'tmp/board-test/local-drivers-ddr-scratch1.json')
        base = report['snapshots'][-1]
        h.check_snapshot(base)
        for field in ('illegal_control_state', 'fatal_bus_error', 'boot_failed'):
            s = copy.deepcopy(base); s['payload']['reason'][field] = True
            with self.assertRaises(RuntimeError): h.check_snapshot(s)
        for section,key in (('ddr','raw_pass'),('levels','ddr_qualified')):
            s = copy.deepcopy(base); s[section][key] = False
            with self.assertRaises(RuntimeError): h.check_snapshot(s)

    def test_supervised_sequence_with_mock_usb_and_programmer(self):
        # Exercise write-once log ownership and every stage without USB. The
        # programmer's own vendor log must not be pre-created by its caller.
        link = Link(); link.idcode = 0x10660a79; link.usb_binding = {'mock': True}
        link.close = lambda: None
        base = h.load(h.ROOT/'tmp/board-test/local-drivers-ddr-scratch1.json')['snapshots'][0]
        url = 'ftdi://0x0403:0x6011:3:7/2'
        class Child:
            def __init__(self, args, **kwargs):
                if '--supervised-program' in args:
                    receipt = Path(args[args.index('--report')+1])
                    assert not receipt.exists() and not receipt.with_suffix('.log').exists()
                    receipt.with_suffix('.log').write_text('MOCK ONLY')
                    item = dict(status='TEMPORARY_JTAG_COMPLETE_PENDING_READ_ONLY_DIAGNOSTICS',
                                bitstream_sha256=h.scratch.BIT_SHA,
                                manifest_sha256=h.scratch.MANIFEST_SHA, url=url)
                else:
                    receipt = Path(args[-2]); item = dict(status='PASS')
                receipt.write_text(json.dumps(item))
            def wait(self, timeout): return 0
        def snapshot(*args):
            s = copy.deepcopy(base)
            for k in ('received_last','accepted_commands','replies'):
                s['live'][k] = (base['live'][k]+len(link.calls)) & 255
            s['live']['received_bits'] = (base['live']['received_bits']+75*len(link.calls)) & 255
            return s
        import fixed_jtag_physical_b
        with tempfile.TemporaryDirectory(dir=h.ROOT/'tmp') as td:
            report = Path(td)/'mock-only.json'; release = Path(td)/'release.json'
            release.write_text('{}')
            with patch.object(h, 'validate_evidence', return_value={}), \
                 patch.object(h.subprocess, 'Popen', Child), \
                 patch.object(h.scratch, 'module', return_value=SimpleNamespace(validated_url=lambda *a:url)), \
                 patch.object(fixed_jtag_physical_b, 'PhysicalBTransport', return_value=link), \
                 patch.object(h.scratch, 'fresh', side_effect=snapshot):
                result = h.sequence(report, release)
            self.assertEqual(result['status'], 'PHYSICAL_ACCEPTED_A_DDR_STAGE_READBACK_PASS', result.get('error'))
            self.assertEqual(result['stage_counts']['writes'], 187504)
            self.assertEqual(result['scratch_counts']['writes'], 2052)
            self.assertFalse(result['cnn_started'])
            self.assertFalse(result['resident_banks_loaded'])
            self.assertEqual(result['unstable_pairs'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
