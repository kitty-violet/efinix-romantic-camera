import copy,unittest
from unittest.mock import patch
import video_service_control as c
from fixed_jtag_loader import validate_payload
from run_units import ROOT

class Bus:
 def __init__(self):self.mem={c.MB:0};self.writes=[];self.corrupt=None;self.fail=False
 def read(self,a):return self.mem.get(a,0)^(int(a==self.corrupt))
 def write(self,a,v):
  self.writes.append((a,v));self.mem[a]=v
  if self.fail:raise TimeoutError('ambiguous write acknowledgement')
  if a==c.MB:self.mem[a]=0

def sample(phase=5):
 return dict(phase=phase,observed_coefficient_crc=c.EXPECTED_CRC['coefficients-padded'],
  apb={'04':(2560<<8)|1,'08':0,'50':c.EXPECTED_CRC['compute'],'54':c.EXPECTED_CRC['dma'],'5c':17,'60':0,'64':c.TAG})

class Tests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.payload=validate_payload(ROOT/'tmp/frame-execution/accepted-a-integration1')
 def test_stale_completion_needs_new_prepared(self):
  b=Bus();r={};samples=[sample(),sample(),sample()]
  with patch.object(c,'snapshot',side_effect=samples):
   c.prepare_once(b,self.payload,r,lambda:None,read_prepared=iter([False,False,True]).__next__,sleep=lambda _:None)
  self.assertEqual(len(r['samples']),2);self.assertEqual(r['status'],'RESIDENT_PREPARED_NOT_RUNNING')
  self.assertEqual(b.writes[-1],(c.MB,c.LOAD));self.assertNotIn((c.MB,c.LIVE),b.writes)
 def test_readback_error_never_sends_load(self):
  b=Bus();b.corrupt=c.MB+16
  with patch.object(c,'snapshot',return_value=sample()),self.assertRaises(IOError):
   c.prepare_once(b,self.payload,{},lambda:None,read_prepared=lambda:False)
  self.assertFalse(any(a==c.MB for a,v in b.writes))
 def test_unchanged_ready_rejects_before_load(self):
  b=Bus()
  with patch.object(c,'snapshot',return_value=sample()),self.assertRaises(RuntimeError):
   c.prepare_once(b,self.payload,{},lambda:None,read_prepared=lambda:True)
  self.assertFalse(any(a==c.MB for a,v in b.writes))
 def test_corrupt_payload_rejects_before_any_write(self):
  b=Bus();data=dict(self.payload);data['dma']=bytes(len(data['dma']))
  with self.assertRaises(ValueError):c.prepare_once(b,data,{},lambda:None,read_prepared=lambda:False)
  self.assertFalse(b.writes)
 def test_ambiguous_live_write_not_retried(self):
  b=Bus();b.fail=True;r={}
  with patch.object(c,'snapshot',return_value=sample()),self.assertRaises(TimeoutError):c.command_once(b,c.LIVE,r,lambda:None,read_prepared=lambda:True)
  self.assertEqual(b.writes,[(c.MB,c.LIVE)]);self.assertTrue(r['command_attempted']);self.assertFalse(r['command_acknowledged'])
 def test_unprepared_or_pending_live_rejected(self):
  for s,pending in ((sample(1),0),(sample(),c.LOAD)):
   b=Bus();b.mem[c.MB]=pending
   with patch.object(c,'snapshot',return_value=s),self.assertRaises(RuntimeError):c.command_once(b,c.LIVE,{},lambda:None)
   self.assertFalse(b.writes)
 def test_load_cannot_compute(self):
  b=Bus();bad=sample();bad['apb']['5c']+=1
  with patch.object(c,'snapshot',side_effect=[sample(),bad]),self.assertRaises(RuntimeError):
   c.prepare_once(b,self.payload,{},lambda:None,read_prepared=lambda:False,sleep=lambda _:None)
 def test_load_bad_resident_crc(self):
  b=Bus();bad=sample();bad['observed_coefficient_crc']=0
  with patch.object(c,'snapshot',side_effect=[sample(),bad]),self.assertRaises(RuntimeError):
   c.prepare_once(b,self.payload,{},lambda:None,read_prepared=iter([False,True]).__next__,sleep=lambda _:None)
 def test_fault_never_submits_run(self):
  b=Bus();bad=sample();bad['phase']=0x80000007
  with patch.object(c,'snapshot',return_value=bad),self.assertRaises(RuntimeError):c.command_once(b,c.LIVE,{},lambda:None)
  self.assertFalse(b.writes)
 def test_modes_no_implicit_start_or_clear(self):
  self.assertEqual(c.diagnostic_word(tx_pattern=True),0x56530001)
  self.assertEqual(c.diagnostic_word(mode=2),0x56530208)
  self.assertEqual(c.diagnostic_word(inspect='result'),0x56530c00)
  with self.assertRaises(ValueError):c.diagnostic_word(inspect='input',clear_window=True)
 def test_stale_ready_cannot_start(self):
  for check in (None,lambda:False):
   b=Bus()
   with patch.object(c,'snapshot',return_value=sample()),self.assertRaises(RuntimeError):
    c.command_once(b,c.LIVE,{},lambda:None,read_prepared=check)
   self.assertFalse(b.writes)

if __name__=='__main__':unittest.main()
