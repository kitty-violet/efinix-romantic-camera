import unittest
from unittest.mock import patch
from read_video_service_pages import decode,PageAssembler
import video_service_observation as o

class CompactTests(unittest.TestCase):
 positions=list(range(192))
 def raw(self,phase,part,*,legacy=False,stale=False,version=4):
  words=[100+i for i in range(16)]
  words[15]={2:0x56494431,3:0x53525631,4:0x484c5431,5:0x50415431}.get(phase,0)
  words[12]=1
  data=sum(w<<(i*32) for i,w in enumerate(words[part*4:part*4+4])) if not legacy else (0 if part==7 else 0x11223344+part)
  tag=(0x4c4f0000|part) if legacy else (0x56530000|part<<3|phase)
  sequence=(part//2)*6+(part%2) if legacy else 17
  packet=data|(tag<<128)|(sequence<<160)|(version<<168)
  if not legacy:packet|=(1<<176)|(int(stale)<<178)
  return ((packet<<64)|0x4f465035|(5<<40)|(1<<32)|(61<<56))<<2
 def test_four_service_pages_preserved(self):
  for phase,kind in ((2,'display'),(3,'process'),(4,'health'),(5,'path')):
   a=PageAssembler()
   for part in range(4):result=a.add(decode(self.raw(phase,part),self.positions,[]))
   self.assertEqual(result['page_kind'],kind);self.assertFalse(result['whole_page_atomic'])
 def test_legacy_reconstruction_exact_with_no_atomic_claim(self):
  a=PageAssembler(camera_slots=[11,22])
  with patch('read_video_service_pages.legacy_decode',return_value={'observed':True}) as legacy:
   for part in range(8):
    result=a.add(decode(self.raw(0,part,legacy=True),self.positions,[]))
    if part<7:self.assertIsNone(result)
  raw,slots=legacy.call_args.args
  expected=sum((0x11223344+i)<<(i*128) for i in range(7))
  self.assertEqual((raw>>2)>>64,expected);self.assertEqual(slots,[11,22])
  self.assertFalse(result['whole_page_atomic']);self.assertEqual(len(result['chunks']),8)
 def test_missing_legacy_chunk_is_not_assembled(self):
  a=PageAssembler()
  for part in (0,1,3,4,5,6,7):self.assertIsNone(a.add(decode(self.raw(0,part,legacy=True),self.positions,[])))
 def test_torn_nested_camera_page_rejected(self):
  for s6,s7 in ((7,8),(15,16),(31,32),(18,21),(18,18)):
   a=PageAssembler()
   with patch('read_video_service_pages.legacy_decode') as legacy:
    for part in range(8):
     c=decode(self.raw(0,part,legacy=True),self.positions,[])
     if part>=6:c['page_sequence']=(s6,s7)[part-6]
     result=a.add(c)
    self.assertIsNone(result);legacy.assert_not_called()
 def test_nested_camera_choices_including_sequence_wrap(self):
  for s6 in (18,42,66,90,254):
   a=PageAssembler()
   with patch('read_video_service_pages.legacy_decode',return_value={}) as legacy:
    for part in range(8):
     c=decode(self.raw(0,part,legacy=True),self.positions,[])
     if part>=6:c['page_sequence']=(s6+part-6)&255
     result=a.add(c)
    self.assertEqual(result['page_kind'],'legacy');legacy.assert_called_once()
 def test_mixed_schema_is_not_assembled(self):
  a=PageAssembler();c=decode(self.raw(3,0),self.positions,[]);a.add(c)
  c=decode(self.raw(3,1),self.positions,[]);c['diagnostic_profile']='video-service-pages-v2'
  self.assertIsNone(a.add(c))
 def test_schema_padding_and_last_word_checked(self):
  for raw in (self.raw(3,1,version=2),self.raw(3,1)|(1<<800),self.raw(0,7,legacy=True)|(1<<(2+64+64))):
   with self.assertRaises(ValueError):decode(raw,self.positions,[])
 def test_same_atomic_prepared_control(self):
  c=decode(self.raw(3,3),self.positions,[])
  with patch.object(o,'fresh',return_value=c):self.assertTrue(o.prepared(None,self.positions,[],[],lambda:None))
 def test_display_stale_keeps_health_readable(self):
  self.assertFalse(decode(self.raw(2,0,stale=True),self.positions,[])['measurement_valid'])
  self.assertTrue(decode(self.raw(4,0,stale=True),self.positions,[])['measurement_valid'])
if __name__=='__main__':unittest.main()
