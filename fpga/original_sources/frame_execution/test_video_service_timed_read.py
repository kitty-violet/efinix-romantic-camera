import unittest
from video_service_timed_read import rate, summarize


class TimedReadTests(unittest.TestCase):
    def rows(self):
        return [dict(count=65530,before_seconds=0,after_seconds=.01),
                dict(count=24,before_seconds=1,after_seconds=1.01)]
    def test_real_width_wrap(self):
        r=rate(self.rows(),'captured'); self.assertEqual(r['events'],30)
        self.assertAlmostEqual(r['rate_bounds_hz'][0],30/1.01)
        self.assertAlmostEqual(r['rate_bounds_hz'][1],30/.99)
    def test_saturation_refused(self):
        rows=self.rows(); rows[1]['count']=65535
        with self.assertRaises(ValueError):rate(rows,'compute_admitted')
    def test_overlap_refused(self):
        rows=self.rows(); rows[1]['before_seconds']=.005
        with self.assertRaises(ValueError):rate(rows,'captured')
    def test_reset_jump_refused(self):
        rows=self.rows(); rows[0]['count']=400; rows[1]['count']=0
        with self.assertRaises(ValueError):rate(rows,'committed')
    def test_zero_events_not_latency(self):
        rows=self.rows(); rows[1]['count']=rows[0]['count']
        self.assertIsNone(rate(rows,'captured')['mean_event_interval_ms_bounds'])
    def report(self):
        chunks=[]
        for t in (0,1):
            for kind,index,value in [('health',13,1<<16),('path',14,0),('display',11,t*15),('health',12,t*30)]:
                chunks.append(dict(page_kind=kind,chunk_index=index,chunk_words=[value],measurement_valid=True,
                    before_seconds=t,after_seconds=t+.01))
        return dict(chunks=chunks,samples=[dict(phase=4,error=0,count=t*15,before_seconds=t,after_seconds=t+.01) for t in (0,1)])
    def test_distinct_time_and_pixel_cache_limits(self):
        r=summarize(self.report())
        self.assertEqual(r['ai_commits']['events'],15)
        self.assertIsNone(r['loss_fraction'])
        self.assertFalse(r['word_rates']['distinct_frames']['host_intervals_bound_value_capture'])
        self.assertTrue(r['word_rates']['source_area_frames']['host_intervals_bound_value_capture'])
    def test_epoch_change_refused(self):
        r=self.report();r['chunks'][4]['chunk_words']=[2<<16]
        with self.assertRaises(ValueError):summarize(r)
    def test_wrong_mode_refused(self):
        r=self.report();r['chunks'][5]['chunk_words']=[1]
        with self.assertRaises(ValueError):summarize(r)
    def test_stopped_refused(self):
        r=self.report();r['samples'][-1]['phase']=5
        with self.assertRaises(ValueError):summarize(r)


if __name__=='__main__':unittest.main()
