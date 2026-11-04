import unittest
from read_control_snapshot import decode,fresh,SnapshotPending,inspect

def packet(sequence=0,valid=1,busy=0,ticks=100,levels=0x7f):
 flags=0 # Control is released and idle.
 value=0x4f425331 | valid<<32 | busy<<33 | busy<<35 | 1<<40 | levels<<48 | sequence<<56
 words=[ticks,0x0403024b,0xf8100000,0x54493630,flags,0x1234abcf]
 for i,w in enumerate(words):value|=w<<(64+i*32)
 return value<<2

class Transport:
 idcode=0x10660a79;usb_binding={'offline':True}
 def __init__(self,values):self.values=iter(values);self.calls=[]
 def scan(self,value,count):self.calls.append((value,count));return next(self.values)

class Tests(unittest.TestCase):
 def test_layout(self):
  r=decode(packet());self.assertEqual(r['payload']['remote_bits'],75);self.assertEqual(r['payload']['host_replies'],4)
  self.assertEqual(r['payload']['last_host_address'],'f8100000');self.assertFalse(r['levels']['control_reset'])
 def test_refuse_wrong_protocol(self):
  for p in (0,packet()^(1<<2),packet()^(1<<42),packet()|(1<<38),-1,1<<258):
   with self.assertRaises(ValueError):decode(p)
 def test_no_valid_payload(self):self.assertIsNone(decode(packet(valid=0))['payload'])
 def test_sequence_wrap_and_stale_poll(self):
  t=Transport([packet(255),packet(255),packet(255,busy=1),packet(0)])
  history=[];r=fresh(t,history,lambda:None)
  self.assertEqual(r['sequence'],0);self.assertEqual(t.calls,[(3,258),(2,258),(3,258),(3,258)])
 def test_clock_stopped_prefix_retained(self):
  t=Transport([packet(0),packet(0)]+[packet(0,busy=1)]*8);history=[]
  with self.assertRaises(SnapshotPending):fresh(t,history,lambda:None)
  self.assertEqual(len(history),10);self.assertTrue(history[-1]['busy'])
 def test_pending_request_not_toggled(self):
  t=Transport([packet(0,busy=1)]*9);history=[]
  with self.assertRaises(SnapshotPending):fresh(t,history,lambda:None)
  self.assertTrue(all(v==3 for v,n in t.calls))
 def test_default_never_issues_command(self):
  t=Transport([packet(0),packet(0),packet(1),packet(1),packet(1),packet(2,ticks=101)])
  result={};inspect(t,result,lambda:None)
  self.assertEqual(result['system_ticks_advanced'],1);self.assertTrue(all(v in (2,3) and n==258 for v,n in t.calls))
 def test_reset_skips_identity(self):
  t=Transport([packet(0),packet(0),packet(1,levels=255),packet(1),packet(1),packet(2,ticks=101,levels=255)])
  result={};inspect(t,result,lambda:None,True);self.assertEqual(result['identity_probe'],'SKIPPED_CLOCK_OR_RESET')
 def probe(self,responses):
  t=Transport([packet(0),packet(0),packet(1),packet(1),packet(1),packet(2,ticks=101),0,*responses,
               packet(2),packet(2),packet(3,ticks=102)])
  result={};inspect(t,result,lambda:None,True)
  self.assertEqual(result['identity_requests'],1)
  self.assertEqual(sum(n==77 for v,n in t.calls),1)
  self.assertEqual(len(result['snapshots']),3)
  self.assertEqual(result['snapshots'][-1]['sequence'],3)
  return result
 def test_identity_success(self):
  r=self.probe([(0x54493630<<4)|4]);self.assertEqual(r['identity_probe'],'54493630')
 def test_identity_error_keeps_independent_observation(self):
  r=self.probe([12]);self.assertIn('bus error',r['identity_error'])
 def test_identity_timeout_keeps_independent_observation(self):
  r=self.probe([0]*200);self.assertIn('timeout',r['identity_error'])
if __name__=='__main__':unittest.main()
