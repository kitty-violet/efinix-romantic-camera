"""Real subprocess deadline regression without importing a USB backend."""
import argparse,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import read_camera_checked_snapshot_v2 as reader

class BudgetTests(unittest.TestCase):
    def run_child(self,source,validation=1.5,io=.3):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[4]/'tmp/board-test') as td:
            report=Path(td)/'report.json';script=Path(td)/'worker.py'
            script.write_text('import json,time\nfrom pathlib import Path\np=Path('+repr(str(report))+')\n'+source)
            return reader.supervise([sys.executable,str(script)],report,Path(td)/'worker.log',validation,io)
    def test_slow_validation_does_not_consume_io_budget(self):
        code,r=self.run_child("time.sleep(.5)\np.write_text(json.dumps(dict(status='READING',hardware_access=True)))\ntime.sleep(.1)\np.write_text(json.dumps(dict(status='DONE',hardware_access=True)))\n")
        self.assertEqual(code,0);self.assertEqual(r['status'],'DONE')
    def test_validation_stall_stops_before_any_hardware(self):
        code,r=self.run_child('time.sleep(10)\n',validation=.3)
        self.assertEqual(code,124);self.assertEqual(r['timeout_phase'],'offline_validation');self.assertFalse(r['hardware_access'])
    def test_io_stall_has_its_own_short_deadline(self):
        code,r=self.run_child("p.write_text(json.dumps(dict(status='READING',hardware_access=True)))\ntime.sleep(10)\n")
        self.assertEqual(code,124);self.assertEqual(r['timeout_phase'],'hardware_io')
        self.assertLess(r['host_seconds'],1.5)
    def test_bad_evidence_cannot_reach_transport(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[4]/'tmp/board-test') as td:
            a=argparse.Namespace(report=Path(td)/'r.json',manifest=Path(td)/'missing.json',url='MUST_NOT_OPEN')
            with patch.object(reader,'checked_manifest',side_effect=ValueError('hash mismatch')):
                self.assertEqual(reader.worker(a),1)
            r=json.loads(a.report.read_text());self.assertFalse(r['hardware_access']);self.assertEqual(r['scans'],[])

if __name__=='__main__':unittest.main()
