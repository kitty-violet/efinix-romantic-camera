import unittest
from unittest.mock import patch
import read_command_progress_status as reader

class Transport:
    def __init__(self,timeout=False):self.timeout=timeout;self.commands=[];self.address=0
    def scan(self,value,count):
        if count==77:
            assert not (value>>74&1),'Read-only test sent write'
            self.address=(value>>10)&0xffffffff;self.commands.append(self.address);return 0
        assert (value,count)==(1,36)
        if self.timeout:return 0
        data={0xf8100000:0x54493630,0xf8100004:0,0xf8100008:0,0xf810000c:0xb1000000,0xf810006c:0x3f}[self.address]
        return ((data<<2)|1)<<2

class Tests(unittest.TestCase):
    def output(self):return dict(status='READING',scans=[],snapshots=[],samples=[])
    def snapshots(self,n=15):
        before={'accepted_commands':250,'replies':250,'received_last':250,'received_bits':250}
        after={name:(value+n*(75 if name=='received_bits' else 1))&255 for name,value in before.items()}
        return [{'levels':{'control_reset':False},'live':live} for live in (before,after)]
    def test_normal(self):
        t=Transport();out=self.output()
        with patch.object(reader,'fresh',side_effect=self.snapshots()):reader.inspect(t,out,lambda:None)
        self.assertEqual(out['status'],'OFP2_AND_STABLE_STATUS_READ_COMPLETE_NOT_BOARD_PASS')
        self.assertEqual(len(out['snapshots']),2);self.assertEqual(out['transactions'],15)
        self.assertEqual(set(t.commands),{0xf8100000,0xf8100004,0xf8100008,0xf810000c,0xf810006c})
    def test_bus_timeout_still_collects_final_snapshot(self):
        out=self.output()
        with patch.object(reader,'fresh',return_value={'levels':{'control_reset':False}}):
            with self.assertRaises(TimeoutError):reader.inspect(Transport(True),out,lambda:None)
        self.assertEqual(len(out['snapshots']),2);self.assertEqual(out['transactions'],0)
    def test_reset_no_command(self):
        out=self.output();t=Transport()
        with patch.object(reader,'fresh',return_value={'levels':{'control_reset':True}}):
            with self.assertRaises(RuntimeError):reader.inspect(t,out,lambda:None)
        self.assertEqual(t.commands,[])
    def test_final_snapshot_failure_never_passes(self):
        out=self.output()
        with patch.object(reader,'fresh',side_effect=[{'levels':{'control_reset':False}},ValueError('bad signature')]):reader.inspect(Transport(),out,lambda:None)
        self.assertEqual(out['status'],'FAIL_PREFIX_SAVED');self.assertIn('bad signature',out['final_snapshot_error'])
    def test_stale_response_rejected_by_progress(self):
        out=self.output()
        with patch.object(reader,'fresh',side_effect=self.snapshots(0)):reader.inspect(Transport(),out,lambda:None)
        self.assertEqual(out['status'],'FAIL_PREFIX_SAVED');self.assertEqual(out['progress_delta']['accepted_commands'],0)
    def test_progress_wrap(self):
        out=self.output()
        with patch.object(reader,'fresh',side_effect=self.snapshots()):reader.inspect(Transport(),out,lambda:None)
        self.assertEqual(out['progress_delta'],dict(accepted_commands=15,replies=15,received_last=15,received_bits=101))

if __name__=='__main__':unittest.main()
