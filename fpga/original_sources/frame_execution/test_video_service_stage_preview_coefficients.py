import json,sys,unittest
from pathlib import Path
from unittest.mock import patch
import video_service_stage_preview_coefficients as c
from fixed_jtag_loader import APB,MB


class Bus:
    def __init__(self):
        self.mem={APB:0x54493630,APB+4:0,APB+8:0,APB+12:0xb1000000,
                  APB+0x6c:255,MB:0,MB+4:1}
        self.writes=[];self.corrupt=None;self.fail=None
    def read(self,a):return self.mem.get(a,0) ^ int(a==self.corrupt)
    def write(self,a,v):
        self.writes.append((a,v))
        if a==self.fail:raise TimeoutError('Ambiguous reply')
        self.mem[a]=v


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.image=(c.ROOT/'model-training-workspace/accepted/B1-A-20261008/coefficients-padded.bin').read_bytes()
    def test_cold_empty_ddr_fully_written_and_read(self):
        b=Bus();out={};c.transfer_coefficients(b,self.image,out,lambda:None)
        self.assertEqual(len(b.writes),5120);self.assertEqual(out['verified_bytes'],20480)
        self.assertEqual(out['readback']['sha256'],c.EXPECTED)
        self.assertTrue(all(c.BASE<=a<c.BASE+c.SIZE for a,v in b.writes))
        self.assertFalse(out['cnn_started']);self.assertEqual(out['mailbox_writes'],0)
    def test_bad_package_rejected_before_writes(self):
        b=Bus()
        with self.assertRaises(ValueError):c.transfer_coefficients(b,bytes(c.SIZE),{},lambda:None)
        self.assertEqual(b.writes,[])
    def test_running_or_failed_loader_rejected(self):
        for address,value in ((APB+8,3),(MB+4,0x80000003),(APB+4,8)):
            b=Bus();b.mem[address]=value
            with self.assertRaises(RuntimeError):c.transfer_coefficients(b,self.image,{},lambda:None)
            self.assertEqual(b.writes,[])
    def test_readback_mismatch_stops_first_word(self):
        b=Bus();b.corrupt=c.BASE+8;out={}
        with self.assertRaises(IOError):c.transfer_coefficients(b,self.image,out,lambda:None)
        self.assertEqual(out['verified_bytes'],12);self.assertEqual(out['mismatch_count'],1)
    def test_ambiguous_write_never_retried(self):
        b=Bus();b.fail=c.BASE+4;out={}
        with self.assertRaises(TimeoutError):c.transfer_coefficients(b,self.image,out,lambda:None)
        self.assertEqual(len(b.writes),2);self.assertEqual(out['written_bytes'],4)
    def test_scope_rejects_program_and_mailbox_before_transport(self):
        with patch.object(c.StageBus,'access') as transport:
            b=c.CoefficientBus(None)
            for address in (0xe1000000,0xe1100000,MB,APB,c.BASE+c.SIZE):
                with self.assertRaises(ValueError):b.access(True,address,0)
            transport.assert_not_called()


if __name__=='__main__':
    target=Path(sys.argv[1]);assert not target.exists()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    target.write_text(json.dumps(dict(status='PASS' if result.wasSuccessful() else 'FAIL',
        tests=result.testsRun,helper_sha256=c.sha(c.__file__),test_sha256=c.sha(__file__)),indent=2),encoding='utf-8')
    raise SystemExit(not result.wasSuccessful())
