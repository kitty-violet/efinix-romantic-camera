import json,unittest
from pathlib import Path
from unittest.mock import patch
from read_video_service_pages import decode,PageAssembler
import video_service_observation as observation

ROOT=Path(__file__).resolve().parents[4]
class WordTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.positions=json.loads((ROOT/'tmp/frame-execution/video-service-native20/graft.json').read_text())['display_extension']['observation_positions']
  assert len(cls.positions)==87
 def raw(self,phase,part,*,stale=False,valid=True,version=3,reserved=0):
  data=100+part
  if part==15:data={2:0x56494431,3:0x53525631,4:0x484c5431,5:0x50415431}[phase]
  if phase==3 and part==12:data=0x53520001
  packet=data|((0x56530000|part<<3|phase)<<32)|((part*6+phase)<<64)|(version<<72)|(int(valid)<<80)|(int(stale)<<82)|(reserved<<83)
  payload=sum(((packet>>i)&1)<<p for i,p in enumerate(self.positions))
  return ((payload<<64)|0x4f465035|(5<<40)|(1<<32)|(61<<56))<<2
 def test_every_field_word_uses_actual_capture_positions(self):
  for phase,kind in ((2,'display'),(3,'process'),(4,'health'),(5,'path')):
   a=PageAssembler()
   for part in range(16):
    c=decode(self.raw(phase,part),self.positions,[])
    self.assertEqual(c['chunk_index'],part);self.assertEqual(c['page_phase'],phase)
    self.assertEqual(c['chunk_bits'],32);self.assertFalse(c['whole_page_atomic'])
    result=a.add(c)
    if part<15:self.assertIsNone(result)
   self.assertEqual(result['page_kind'],kind);self.assertEqual(len(result['chunks']),16)
   self.assertFalse(result['whole_page_atomic'])
   if phase==3:self.assertEqual(result['fields']['committed'],107)
 def test_missing_word_is_not_complete(self):
  a=PageAssembler()
  for part in (0,1,2,4,5,6,7,8,9,10,11,12,13,14,15):self.assertIsNone(a.add(decode(self.raw(3,part),self.positions,[])))
 def test_wrong_version_and_reserved_rejected(self):
  for args in ({'version':2},{'reserved':1}):
   with self.assertRaises(ValueError):decode(self.raw(3,12,**args),self.positions,[])
 def test_stale_display_does_not_invalidate_process_word(self):
  self.assertFalse(decode(self.raw(2,5,stale=True),self.positions,[])['measurement_valid'])
  self.assertTrue(decode(self.raw(3,12,stale=True,valid=False),self.positions,[])['measurement_valid'])
 def test_prepared_uses_one_fresh_signed_word(self):
  chunk=decode(self.raw(3,12),self.positions,[])
  with patch.object(observation,'fresh',side_effect=[dict(page_kind='legacy')]*95+[chunk]) as f:
   self.assertTrue(observation.prepared(None,self.positions,[],[],lambda:None))
  self.assertEqual(f.call_count,96)
  chunk['chunk_words'][0]&=~1
  with patch.object(observation,'fresh',return_value=chunk):self.assertFalse(observation.prepared(None,self.positions,[],[],lambda:None))
 def test_bad_prepared_signature_cannot_start(self):
  chunk=decode(self.raw(3,12),self.positions,[])
  for bad in (1,0x5352e001):
   chunk['chunk_words']=[bad]
   with patch.object(observation,'fresh',return_value=chunk),self.assertRaises(ValueError):observation.prepared(None,self.positions,[],[],lambda:None)
 def test_missing_prepared_is_bounded(self):
  with patch.object(observation,'fresh',return_value=dict(page_kind='legacy')) as f,self.assertRaises(observation.SnapshotPending):
   observation.prepared(None,self.positions,[],[],lambda:None)
  self.assertEqual(f.call_count,96)
if __name__=='__main__':unittest.main()
