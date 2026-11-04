import copy,json,sys,unittest
from pathlib import Path
from unittest.mock import patch
import video_service_input_isp_prepared as c
from test_video_service_input_isp import Tests as ConfigureTests

class GuardTests(unittest.TestCase):
 def fixture(self):
  s={'phase':5,'apb':{'00':0x54493630,'04':0xa0005,'08':0,'60':0,'64':0xa008},'observed_coefficient_crc':0x7c6149bf}
  values={c.MB:0,c.APB+12:0xb1000000,c.APB+0x6c:255}
  class Bus:
   def read(self,a):return values[a]
  return s,values,Bus()
 def check(self,change,expected):
  s,v,b=self.fixture();change(s,v)
  with patch.object(c,'snapshot',return_value=s):
   if expected:c.prepared_guard(b)
   else:
    with self.assertRaises(ValueError):c.prepared_guard(b)
 def test_loaded_stopped_accepts(self):self.check(lambda s,v:None,True)
 def test_cold_unloaded_rejects(self):self.check(lambda s,v:s.update(phase=1),False)
 def test_live_and_fault_reject(self):
  for phase in (4,0x80000007):self.check(lambda s,v:s.update(phase=phase),False)
 def test_ddr_loss_reject(self):
  for d in (254,0x800000ff):self.check(lambda s,v:v.update({c.APB+0x6c:d}),False)
 def test_coeff_error_and_length_reject(self):
  for status in (0xa000d,0x9ff05,0):self.check(lambda s,v:s['apb'].update({'04':status}),False)
 def test_other_weight_identity_reject(self):self.check(lambda s,v:s.update(observed_coefficient_crc=0),False)
 def test_mailbox_pipeline_apb_reject(self):
  self.check(lambda s,v:v.update({c.MB:1}),False)
  self.check(lambda s,v:v.update({c.APB+12:0xb1000001}),False)
  self.check(lambda s,v:s['apb'].update({'60':1}),False)
 def test_configure_is_same_frozen_logic(self):
  import inspect,video_service_input_isp as old
  self.assertEqual(inspect.getsource(c.configure),inspect.getsource(old.configure))
  self.assertEqual(inspect.getsource(c.decode_status),inspect.getsource(old.decode_status))

if __name__=='__main__':
 target=Path(sys.argv[1]);assert not target.exists()
 suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(GuardTests),unittest.defaultTestLoader.loadTestsFromTestCase(ConfigureTests)])
 result=unittest.TextTestRunner(verbosity=2).run(suite)
 target.write_text(json.dumps(dict(status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,helper_sha256=c.sha(c.__file__),test_sha256=c.sha(__file__)),indent=2),encoding='utf-8')
 raise SystemExit(not result.wasSuccessful())
