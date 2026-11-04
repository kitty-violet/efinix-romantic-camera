"""Decode real serial RTL receipts; reject stale, corrupt and legacy scans."""
import re,unittest
from unittest.mock import patch
from run_units import ROOT
from read_control_ddr_progress import decode,fresh,SCAN_BITS
from read_control_snapshot import SnapshotPending

class ControlDdrHost(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        log=(ROOT/'tmp/frame-execution/control-ddr-combined-rtl1/simulation.log').read_text()
        cls.raw=int(re.findall(r'PROGRESSRAW ([0-9a-fA-F]+)',log)[0],16)
    def test_real_rtl_receipt(self):
        d=decode(self.raw);self.assertEqual(d['live']['accepted_commands'],0x34)
        self.assertEqual(d['payload']['address'],'f8100000')
        self.assertTrue(d['ddr']['raw_done']);self.assertTrue(d['ddr']['raw_pass'])
        self.assertEqual(d['ddr']['failure_log'],0xaa)
        self.assertEqual(d['ddr']['calibration_shift_value'],5)
        self.assertEqual(d['ddr']['sampled_shift_requests'],1)
        self.assertNotIn('cal_done_raw',d['levels'])
    def test_reject_invalid(self):
        for raw in (-1,True,1<<SCAN_BITS,self.raw^4,self.raw^(1<<42),self.raw|(1<<38),self.raw|(1<<319)):
            with self.assertRaises(ValueError):decode(raw)
    def test_stale(self):
        source=self
        class Frozen:
            def scan(self,s,n):
                source.assertIn(s,(2,3));source.assertEqual(n,450);return source.raw
        with self.assertRaises(SnapshotPending):fresh(Frozen(),[],lambda:None)
    def test_sequence_wrap(self):
        source=self;history=[]
        class Wrap:
            seq=255
            def scan(self,s,n):
                source.assertEqual(n,450);raw=(source.raw&~(255<<58))|(self.seq<<58)
                if s==2:self.seq=0
                return raw
        self.assertEqual(fresh(Wrap(),history,lambda:None)['sequence'],0)
        self.assertEqual([x['selector'] for x in history],[3,2,3])
    def test_status_reader_uses_ofp3(self):
        import read_control_ddr_status as reader
        from test_command_progress_status import Transport,Tests
        out=Tests().output();t=Transport()
        self.assertEqual(reader.fresh.__module__,'read_control_ddr_progress')
        with patch.object(reader,'fresh',side_effect=Tests().snapshots()):reader.inspect(t,out,lambda:None)
        self.assertEqual(out['status'],'OFP3_AND_STABLE_STATUS_READ_COMPLETE_NOT_BOARD_PASS')
        self.assertEqual(out['transactions'],15)
    def test_status_timeout_retains_independent_snapshot(self):
        import read_control_ddr_status as reader
        from test_command_progress_status import Transport,Tests
        out=Tests().output()
        with patch.object(reader,'fresh',return_value={'levels':{'control_reset':False}}):
            with self.assertRaises(TimeoutError):reader.inspect(Transport(True),out,lambda:None)
        self.assertEqual(len(out['snapshots']),2)
if __name__=='__main__':unittest.main()
