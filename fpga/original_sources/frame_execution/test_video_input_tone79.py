import argparse
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import video_input_tone79 as curve
import video_service_tone79 as runner
from video_input_isp_control import table as old_table, upload_plan
from test_video_service_input_isp import Tests as ProtocolTests
from run_units import ROOT, sha


class CurveTests(unittest.TestCase):
    def test_knots_and_finite_range(self):
        for x, y in curve.KNOTS:self.assertAlmostEqual(curve.tone(x), y)
        for v in (-1,256,math.nan,math.inf):
            with self.assertRaises(ValueError):curve.tone(v)

    def test_shadow_pedestal_and_highlight_reduction(self):
        import video_input_tone78 as previous
        for value in range(1,256):
            self.assertLess(curve.tone(value), previous.tone(value))
        # A lower shadow level with steeper separation in the working shadow/mid band.
        for lo,hi in ((64,96),(96,128)):
            self.assertGreater(curve.tone(hi)-curve.tone(lo), previous.tone(hi)-previous.tone(lo))
        self.assertEqual(curve.tone(0),0)
        self.assertEqual(curve.tone(96),100)
        self.assertEqual(curve.tone(255),204)

    def test_all_codes_monotonic_and_endpoint(self):
        data=curve.table();self.assertEqual(len(data),768)
        curve.validate(data,curve.describe())
        for c in range(3):
            a=data[c*256:(c+1)*256]
            self.assertEqual(a[0],0);self.assertGreater(a[1],0)
            self.assertLessEqual(max(a),204)
            self.assertEqual(list(a),sorted(a))

    def test_no_double_rounding_and_no_early_gain_clip(self):
        # Independent explicit segment interpolation for all768 entries.
        expected=[]
        for gain in curve.GAINS:
            for x in range(256):
                y=255*math.pow(x*gain/255,1/3)
                for i in range(1,len(curve.KNOTS)):
                    hi,hy=curve.KNOTS[i];lo,ly=curve.KNOTS[i-1]
                    if y<=hi:
                        expected.append(math.floor(((hi-y)*ly+(y-lo)*hy)/(hi-lo)+.5));break
        self.assertEqual(curve.table(),bytes(expected))

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

    def test_no_regression_of_black_dead_zone(self):
        import video_input_tone78 as previous
        old=previous.table();new=curve.table()
        self.assertTrue(all(a<=b for a,b in zip(new,old)))
        for c in range(3):
            channel=new[c*256:(c+1)*256]
            self.assertTrue(all(x>0 for x in channel[1:]))
            self.assertLess(channel[-1],old[c*256+255])



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
