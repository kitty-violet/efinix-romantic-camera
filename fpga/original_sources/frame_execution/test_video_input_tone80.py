import argparse
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import video_input_tone80 as curve
import video_service_tone80 as runner
from video_input_isp_control import table as old_table, upload_plan
from test_video_service_input_isp import Tests as ProtocolTests
from run_units import ROOT, sha


class CurveTests(unittest.TestCase):
    def test_neutral_anchors_and_finite_range(self):
        for c,a in enumerate(curve.ANCHORS):
            self.assertEqual(curve.shadow(0,c),0)
            self.assertEqual(curve.shadow(a,c),68)
            self.assertEqual(curve.shadow(a/2,c),20)
            self.assertEqual(curve.shadow(160,c),160)
        for v in (-1,256,math.nan,math.inf):
            with self.assertRaises(ValueError):curve.shadow(v,0)

    def test_highlight_preserved_all_codes(self):
        old=curve.base.table();new=curve.table()
        for a,b in zip(old,new):
            if a>=160:self.assertEqual(a,b)
            else:self.assertLessEqual(b,a)

    def test_all_codes_monotonic_and_endpoint(self):
        data=curve.table();self.assertEqual(len(data),768)
        curve.validate(data,curve.describe())
        for c in range(3):
            a=data[c*256:(c+1)*256]
            self.assertEqual(a[0],0);self.assertGreater(a[1],0)
            self.assertLessEqual(max(a),204)
            self.assertEqual(list(a),sorted(a))

    def test_composition_and_rgb_order(self):
        previous=curve.base.table()
        for c,gain in enumerate(curve.GAINS):
            for x in range(256):
                old=previous[c*256+x]
                v=curve.base.tone(255*math.pow(x*gain/255,1/3))
                a=curve.ANCHORS[c]
                if v<=a/2:y=40*v/a
                elif v<=a:y=20+96*(v-a/2)/a
                elif v<160:y=68+92*(v-a)/(160-a)
                else:y=v
                expected=old if old>=160 else math.floor(y+.5)
                self.assertEqual(curve.table()[c*256+x],expected)

    def test_corruption_and_stale_plan_rejected(self):
        data=bytearray(curve.table());data[100]^=1
        with self.assertRaises(ValueError):curve.validate(bytes(data),curve.describe())
        p=curve.describe();p['black']=8
        with self.assertRaises(ValueError):curve.validate(curve.table(),p)
        with self.assertRaises(ValueError):curve.validate(curve.table()[:-1],curve.describe())

    def test_actual_register_protocol_with_new_data(self):
        plan=upload_plan(curve.table(),True);report={};writes=[]
        class Bus:
            def write(self,address,value):
                self_outer.assertEqual(address,runner.base.MB+0x28);writes.append(value)
            def read(self,address):return writes[-1]
        self_outer=self
        def status(i):
            return dict(valid=True,updating=False,count=768,accepted_crc32=plan['crc32'],swap_rb=True,
                        read_pending=False,read_request=False,rejected=0,read_index=i or 0,
                        read_rgb=plan['readback_rgb'][i or 0])
        runner.base.configure(Bus(),plan,report,lambda:None,check_stopped=lambda:None,read_status=status)
        self.assertEqual(report['readbacks'],plan['readback_rgb'])
        self.assertEqual(writes,plan['words']+[0x4a000000+i for i in range(256)])

    def test_wrapper_uses_exact_frozen_worker(self):
        data=curve.table();a=SimpleNamespace(curve_root=Path('unused'),test_receipt=Path('base'),curve_test_receipt=Path('new'))
        with patch.object(runner,'payload',return_value=data), patch.object(runner.base,'verified_tests'), \
             patch.object(runner,'verify_tests'), patch.object(runner.base,'worker') as worker:
            runner.run_worker(a)
            worker.assert_called_once_with(a,upload_plan(data,True))
            self.assertEqual(a.data,data)

    def test_file_scope_and_payload(self):
        with self.assertRaises(ValueError):runner.payload(Path('C:/outside-curve'))
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp/board-test') as d:
            p=Path(d);(p/'plan.json').write_text(json.dumps(curve.describe()))
            (p/'table.bin').write_bytes(curve.table())
            self.assertEqual(runner.payload(p),curve.table())
            (p/'table.bin').write_bytes(bytes(768))
            with self.assertRaises(ValueError):runner.payload(p)

    def test_no_new_black_dead_zone_and_neutral_sample_source(self):
        data=curve.table()
        for c in range(3):self.assertTrue(all(v>0 for v in data[c*256+1:(c+1)*256]))
        sample=ROOT/'tmp/board-test/video-exposure76-board1/curve79/dark-neutral-row1.frame-prefix.bin'
        self.assertEqual(sha(sample),curve.SAMPLE_SHA)
        data=bytes(v^128 for v in sample.read_bytes())
        self.assertEqual(tuple(sum(data[c::3])/256 for c in range(3)),curve.ANCHORS)



if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',required=True,type=Path);a=p.parse_args()
    assert not a.report.exists()
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(CurveTests),
                             unittest.defaultTestLoader.loadTestsFromTestCase(ProtocolTests)])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    a.report.write_text(json.dumps(dict(status='PASS' if result.wasSuccessful() else 'FAIL',
        tests=result.testsRun,runner_sha256=sha(runner.__file__),generator_sha256=sha(curve.__file__),
        worker_sha256=sha(runner.base.__file__),test_sha256=sha(__file__)),indent=2),encoding='utf-8')
    raise SystemExit(not result.wasSuccessful())
