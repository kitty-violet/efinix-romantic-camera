import unittest
from video_service_drained_receipt import extract


class DrainedReceiptTests(unittest.TestCase):
    def sample(self):
        process=dict(status=48,recovery_status=1,owner_states=[0,6,5],captured=100,compute_admitted=90,committed=90)
        r=dict(status='READ_ONLY_SERVICE_OBSERVED_NOT_BOARD_PASS',manifest_sha256='manifest',
            register_samples=[dict(phase=5,apb={'08':0,'60':0})],control=dict(errors=0),pages=[])
        for _ in range(2):
            for k,v in [('process',process.copy()),('health',dict(reset_epochs=0,first_faults=1)),('path',dict(mode='LIVE_A'))]:
                r['pages'].append(dict(page_kind=k,fields=v))
        return r,dict(manifest_sha256='manifest',bitstream_sha256='bit')
    def test_old_raw_fault_does_not_erase_valid_counts(self):
        r,m=self.sample();self.assertEqual(extract(r,m)['captured'],100)
    def test_active_frame_refused(self):
        r,m=self.sample();r['pages'][3]['fields']['owner_states']=[0,3,5]
        with self.assertRaises(ValueError):extract(r,m)
    def test_counts_not_stable_refused(self):
        r,m=self.sample();r['pages'][3]['fields']['committed']=91
        with self.assertRaises(ValueError):extract(r,m)
    def test_status_inflight_refused(self):
        r,m=self.sample();r['pages'][3]['fields']['status']|=64
        with self.assertRaises(ValueError):extract(r,m)
    def test_not_stopped_refused(self):
        r,m=self.sample();r['register_samples'][-1]['phase']=4
        with self.assertRaises(ValueError):extract(r,m)
    def test_wrong_release_refused(self):
        r,m=self.sample();m['manifest_sha256']='other'
        with self.assertRaises(ValueError):extract(r,m)
    def test_actual_failed_bit_refused(self):
        r,m=self.sample();r['pages'][3]['fields']['recovery_status']|=1<<5
        with self.assertRaises(ValueError):extract(r,m)
    def test_epoch_change_refused(self):
        r,m=self.sample();r['pages'][4]['fields']['reset_epochs']=1
        with self.assertRaises(ValueError):extract(r,m)


if __name__=='__main__':unittest.main()
