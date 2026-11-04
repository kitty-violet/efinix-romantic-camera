import copy, json, sys, unittest
from pathlib import Path
import video_service_reference_sample as c


class Tests(unittest.TestCase):
    def tearDown(self):
        c.OFFSET_X=c.OFFSET_Y=0
    def test_all_four_grid_phases(self):
        all_addresses=[]
        for x,y in [(0,0),(8,0),(0,8),(8,8)]:
            c.OFFSET_X,c.OFFSET_Y=x,y
            a=c.addresses()
            self.assertEqual(a[0],c.BASE+3*(1280*y+x))
            all_addresses.extend(a)
        self.assertEqual(len(set(all_addresses)),160*90)
        self.assertTrue(all(a%4==0 and c.BASE<=a<c.BASE+1280*720*3-3 for a in all_addresses))
    def test_invalid_phase_rejected(self):
        for x,y in [(-8,0),(4,0),(0,16),(0,-8)]:
            c.OFFSET_X,c.OFFSET_Y=x,y
            with self.assertRaises(ValueError): c.addresses()

    def fixture(self):
        return {'snapshot': {'phase':5,'observed_coefficient_crc':0x7c6149bf,'apb':{'5c':17283}}}, {'mode':1,'inspect_enabled':True,'inspect_result':False,'inspect_bank':2}
    def test_address_order_and_bounds(self):
        a=c.addresses(); self.assertEqual(len(a),3600); self.assertEqual(len(set(a)),3600)
        self.assertEqual(a[0],0xe1000000);self.assertEqual(a[80],0xe1000000+16*1280*3)
        self.assertTrue(all(x%4==0 and 0xe1000000<=x<0xe1000000+1280*720*3 for x in a))
    def test_q8_channel_order(self):
        self.assertEqual(c.rgb(0x997f0080),bytes([0,128,255]))
    def test_invalid_word(self):
        for x in [-1,1<<32,None,True]:
            with self.assertRaises(ValueError):c.rgb(x)
    def test_lock_policy(self):
        _,p=self.fixture();self.assertEqual(c.locked(p),(1,True,False,2))
        for k,v in [('mode',0),('inspect_enabled',False),('inspect_result',True),('inspect_bank',3)]:
            bad=dict(p);bad[k]=v
            with self.assertRaises(ValueError):c.locked(bad)
    def test_stable_capture(self):
        s,p=self.fixture();c.same_capture(s,copy.deepcopy(s),p,dict(p,chunk_sequence=15))
    def test_changed_bank_rejects(self):
        s,p=self.fixture()
        with self.assertRaises(ValueError):c.same_capture(s,s,p,dict(p,inspect_bank=1))
    def test_reset_counter_rejects(self):
        s,p=self.fixture();t=copy.deepcopy(s);t['snapshot']['apb']['5c']=0
        with self.assertRaises(ValueError):c.same_capture(s,t,p,p)
    def test_read_failure_no_replay(self):
        seen=[]
        class Bus:
            def read(self,a):
                seen.append(a)
                if len(seen)==9:raise TimeoutError('Injected failure')
                return 0x80808080
        out={}
        with self.assertRaises(TimeoutError):c.read_samples(Bus(),out,lambda:None)
        self.assertEqual(seen,c.addresses()[:9]);self.assertEqual(out['pixels_read'],8)
    def test_complete_grid_payload(self):
        seen=[]
        class Bus:
            def read(self,a):seen.append(a);return 0x997f0080
        out={};data=c.read_samples(Bus(),out,lambda:None)
        self.assertEqual(seen,c.addresses());self.assertEqual(data,bytes([0,128,255])*3600)


if __name__=='__main__':
    target=Path(sys.argv[1]);assert not target.exists()
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    target.write_text(json.dumps(dict(status='PASS' if r.wasSuccessful() else 'FAIL', tests=r.testsRun,helper_sha256=c.sha(c.__file__),test_sha256=c.sha(__file__))),encoding='utf-8')
    raise SystemExit(not r.wasSuccessful())
