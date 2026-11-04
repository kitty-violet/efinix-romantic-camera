import unittest
from unittest.mock import patch
from video_service_frame_budget import budget


class FrameBudgetTest(unittest.TestCase):
    def setUp(self):
        self.report={'register_samples':[{'phase':5,'apb':{'08':0,'60':0,'58':9_500_000,'5c':7}} for _ in range(2)]}
        self.manifest={'frequency_mhz':150}
        self.boundary={'committed':7,'reset_epoch':0}
        self.mock=patch('video_service_frame_budget.extract',return_value=self.boundary)
        self.mock.start();self.addCleanup(self.mock.stop)

    def test_stopped_budget(self):
        r=budget(self.report,self.manifest)
        self.assertAlmostEqual(r['issue_active_ms'],63.333333333)
        self.assertEqual(r['issue_budget_headroom_cycles'],500_000)
        self.assertFalse(r['proves_sustained_15fps'])

    def test_active_is_not_full_frame(self):
        self.report['register_samples'][0]['phase']=4
        with self.assertRaises(ValueError):budget(self.report,self.manifest)

    def test_changed_counter_rejected(self):
        self.report['register_samples'][1]['apb']['58']+=1
        with self.assertRaises(ValueError):budget(self.report,self.manifest)

    def test_commit_mismatch_rejected(self):
        self.boundary['committed']=8
        with self.assertRaises(ValueError):budget(self.report,self.manifest)

    def test_missing_frame_rejected(self):
        self.boundary['committed']=0
        with self.assertRaises(ValueError):budget(self.report,self.manifest)

    def test_upstream_drain_failure_propagated(self):
        with patch('video_service_frame_budget.extract',side_effect=ValueError('not drained')):
            with self.assertRaises(ValueError):budget(self.report,self.manifest)


if __name__=='__main__':unittest.main()
