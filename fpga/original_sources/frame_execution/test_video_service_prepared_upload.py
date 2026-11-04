"""Real-sized prepared transfer, rejected states and no-repeat failure checks."""
import json
import sys
import unittest
from pathlib import Path
import video_service_prepared_upload as h
from fixed_jtag_loader import APB, MB
from test_ddr_scratch_bringup import Link
from test_accepted_a_ddr_stage import AliasLink, CorruptLink
from test_video_service_input_isp_prepared import GuardTests


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = h.stage.validate_payload(h.stage.INTEGRATION)

    def transfer(self, link, data=None):
        link.mem.update({APB+4:0xa0005, MB+28:0x7c6149bf, APB+0x64:0xa008, MB+4:5})
        stable = h.stage.StableReplyTransport(link)
        control = h.ControlBus(stable, seconds=100)
        self.bus = h.stage.StageBus(stable, seconds=100)
        self.out = dict(images=[],guards=[])
        h.transfer_checked(self.bus, self.data if data is None else data, self.out, lambda:None,
                           guard=lambda:h.prepared_guard(control))

    def test_real_three_images_roundtrip(self):
        link=Link(); original=h.stage.scratch.guard
        self.transfer(link)
        self.assertIs(h.stage.scratch.guard,original)
        self.assertEqual(self.bus.writes,187504)
        self.assertEqual(self.bus.reads,187504)
        self.assertEqual([r['status'] for r in self.out['images']],['PASS']*3)
        self.assertEqual(sum(r['verified_bytes'] for r in self.out['images']),750016)
        writes=[i for i,(w,a,v) in enumerate(link.calls) if w]
        reads=[i for i,(w,a,v) in enumerate(link.calls) if not w and any(b<=a<b+n for b,n in h.stage.WINDOWS.values())]
        self.assertGreater(min(reads),max(writes))
        self.assertTrue(all(any(b<=a<b+n for b,n in h.stage.WINDOWS.values()) for w,a,v in link.calls if w))

    def test_cross_region_alias(self):
        with self.assertRaises(IOError):self.transfer(AliasLink())
        self.assertTrue(all(r['written_bytes']==r['bytes'] for r in self.out['images']))

    def test_corruption_first_word(self):
        with self.assertRaises(IOError):self.transfer(CorruptLink())
        self.assertEqual(self.out['images'][0]['first_mismatch']['address'],'e1200004')

    def test_wrong_payload_before_any_scan(self):
        raw=bytearray(self.data['compute']);raw[0]^=1
        link=Link()
        with self.assertRaises(ValueError):self.transfer(link,dict(self.data,compute=raw))
        self.assertEqual(link.calls,[])

    def test_live_ddr_mailbox_fault_before_any_write(self):
        for a,v in ((APB+8,3),(APB+0x6c,0),(MB,1),(APB+0x60,1)):
            link=Link();link.mem[a]=v
            with self.assertRaises(ValueError):self.transfer(link)
            self.assertFalse(any(w for w,a,v in link.calls))

    def test_error_no_repeated_writes(self):
        link=Link('timeout'); bus=h.stage.StageBus(h.stage.StableReplyTransport(link))
        with self.assertRaises(TimeoutError):bus.write(0xe1200000,1)
        self.assertEqual(len(link.calls),1)

    def test_old_guard_still_rejects_loaded(self):
        link=Link();link.mem[APB+4]=0xa0005
        with self.assertRaises(RuntimeError):h.stage.scratch.guard(h.stage.scratch.ScratchBus(link))

    def test_drained_inspection_epoch(self):
        good=dict(status=48,owners=5<<6,path=1,epoch=0)
        h.require_idle_words(good)
        for k,values in [('status',[0,49,112]),('owners',[1,2,3]),('path',[5,2,64]),('epoch',[65535])]:
            for v in values:
                with self.assertRaises(ValueError):h.require_idle_words(dict(good,**{k:v}))


if __name__=='__main__':
    path=Path(sys.argv[1]); assert not path.exists()
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (TransferTests,GuardTests)])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    files=[Path(h.__file__),Path(__file__),Path(h.stage.__file__),Path(__file__).with_name('video_service_input_isp_prepared.py')]
    path.write_text(json.dumps(dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
        sources={str(p.resolve()):h.sha(p) for p in files}),indent=2),encoding='utf-8')
    raise SystemExit(not result.wasSuccessful())
