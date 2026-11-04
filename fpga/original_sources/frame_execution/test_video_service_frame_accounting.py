"""Counter rollover, saturation, drain, reset and denominator regressions."""
import copy,unittest
from video_service_frame_accounting import delta,drained_batch

class AccountingTests(unittest.TestCase):
    def base(self):
        return dict(phase=5,mode='LIVE_A',work_drained=True,apb_errors=0,owner_error=False,
                    control_errors=0,stable_process_counts=True,reset_epoch=1,bitstream_sha256='same',
                    captured=65530,compute_admitted=200,committed=0xfffffff0)
    def pair(self):
        a=self.base();b={**a,'captured':14,'compute_admitted':217,'committed':0}
        return a,b
    def test_wrap_and_conservation(self):
        a,b=self.pair();r=drained_batch(a,b)
        self.assertEqual((r['accepted_input_frames'],r['capture_failed_after_admission'],r['cnn_failed'],r['cnn_completed']),(20,3,1,16))
        self.assertEqual(r['admitted_frame_failure_fraction'],0.2)
        self.assertIsNone(r['raw_failures_before_admission'])
    def test_no_events_not_zero_fps_claim(self):
        a=self.base();r=drained_batch(a,a);self.assertIsNone(r['admitted_frame_failure_fraction']);self.assertNotIn('fps',r)
    def test_16_bit_overflow_not_32_bit(self):self.assertEqual(delta('captured',65535,2,10),3)
    def test_saturation_is_unknown(self):
        for x,y in [(65534,65535),(65535,65535)]:
            with self.assertRaises(ValueError):delta('compute_admitted',x,y,100)
    def test_counter_reset(self):
        with self.assertRaises(ValueError):delta('compute_admitted',200,2,100)
    def test_unknown_wrap_count(self):
        with self.assertRaises(ValueError):delta('captured',1,1,65536)
    def test_discontinuity(self):
        with self.assertRaises(ValueError):delta('committed',200,1,1000)
    def test_pending_frame_not_loss(self):
        a,b=self.pair();b['work_drained']=False
        with self.assertRaises(ValueError):drained_batch(a,b)
    def test_reset_epoch_changed(self):
        a,b=self.pair();b['reset_epoch']=2
        with self.assertRaises(ValueError):drained_batch(a,b)
    def test_saturated_reset_history_cannot_prove_same_epoch(self):
        a,b=self.pair();a['reset_epoch']=b['reset_epoch']=65535
        with self.assertRaises(ValueError):drained_batch(a,b)
    def test_wrong_mode(self):
        a,b=self.pair();b['mode']='INPUT_PREVIEW'
        with self.assertRaises(ValueError):drained_batch(a,b)
    def test_mixed_time_counts_rejected(self):
        a,b=self.pair();b['stable_process_counts']=False
        with self.assertRaises(ValueError):drained_batch(a,b)
    def test_bad_balance(self):
        a,b=self.pair();b['compute_admitted']=225
        with self.assertRaises(ValueError):drained_batch(a,b)
    def test_fault_rejected(self):
        a,b=self.pair();b['owner_error']=True
        with self.assertRaises(ValueError):drained_batch(a,b)
    def test_width(self):
        with self.assertRaises(ValueError):delta('captured',0,65536,100)

if __name__=='__main__':unittest.main()
