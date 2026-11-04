import unittest
from unittest.mock import patch
import video_service_observation as o

class Transport:
 def __init__(self):self.seq=0;self.calls=[]
 def scan(self,selector,bits):
  self.calls.append((selector,bits));v=self.seq
  if selector==2:self.seq=(self.seq+1)&255
  return v

def row(seq):return dict(sequence=seq,busy=False,valid=True)

class Tests(unittest.TestCase):
 def test_request_once_wrap(self):
  t=Transport();t.seq=255;history=[]
  with patch.object(o,'decode',side_effect=lambda raw,*args:row(raw)):
   got=o.fresh(t,[],[],history,lambda:None)
  self.assertEqual(got['sequence'],0);self.assertEqual(t.calls,[(3,1026),(2,1026),(3,1026)])
 def test_pending_has_no_new_request(self):
  t=Transport()
  with patch.object(o,'decode',return_value=dict(sequence=3,busy=True,valid=True)),self.assertRaises(o.SnapshotPending):
   o.fresh(t,[],[],[],lambda:None)
  self.assertEqual(len(t.calls),9);self.assertTrue(all(s==3 for s,b in t.calls))
 def test_no_fresh_completion_does_not_replay(self):
  t=Transport()
  with patch.object(o,'decode',return_value=row(3)),self.assertRaises(o.SnapshotPending):o.fresh(t,[],[],[],lambda:None)
  self.assertEqual(sum(s==2 for s,b in t.calls),1)
 def test_full_sweep_and_unprepared(self):
  for prepared in (False,True):
   chunks=[dict(page_kind='legacy')]*23+[dict(page_kind='process',chunk_index=3,measurement_valid=True,chunk_atomic=True,chunk_words=[int(prepared),9,11,0x53525631])]
   with patch.object(o,'fresh',side_effect=chunks) as f:
    self.assertEqual(o.prepared(None,[],[],[],lambda:None),prepared)
   self.assertEqual(f.call_count,24)
 def test_wrong_signature_never_ready(self):
  chunk=dict(page_kind='process',chunk_index=3,measurement_valid=True,chunk_atomic=True,chunk_words=[1,9,11,0])
  with patch.object(o,'fresh',return_value=chunk),self.assertRaises(ValueError):o.prepared(None,[],[],[],lambda:None)
 def test_missing_page_bounded(self):
  with patch.object(o,'fresh',return_value=dict(page_kind='legacy')) as f,self.assertRaises(o.SnapshotPending):o.prepared(None,[],[],[],lambda:None)
  self.assertEqual(f.call_count,24)
 def test_v6_ready_at_end_of_112_chunk_sweep(self):
  chunk=dict(page_kind='process',chunk_index=12,measurement_valid=True,chunk_atomic=True,chunk_words=[1],diagnostic_profile='video-service-pages-v6')
  with patch.object(o,'fresh',side_effect=[dict(page_kind='legacy')]*111+[chunk]) as f:
   self.assertTrue(o.prepared(None,list(range(96)),[],[],lambda:None))
  self.assertEqual(f.call_count,112)
 def test_v6_path_status(self):
  chunk=dict(page_kind='path',chunk_index=14,measurement_valid=True,chunk_words=[0x19],diagnostic_profile='video-service-pages-v6',sequence=42)
  self.assertEqual(o.path_status(chunk),dict(mode=1,inspect_enabled=False,inspect_result=True,inspect_bank=1,chunk_sequence=42))
  chunk['chunk_words']=[0x100];self.assertRaises(ValueError,o.path_status,chunk)

if __name__=='__main__':unittest.main()
