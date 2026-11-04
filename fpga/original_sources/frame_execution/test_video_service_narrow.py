import unittest
from unittest.mock import patch
from read_video_service_pages import decode,PageAssembler
import video_service_observation as o

class NarrowTests(unittest.TestCase):
 positions=list(range(96))
 def raw(self,phase,part,*,legacy=False,stale=False,version=5,sequence=247,word=None):
  if word is None:
   word=(0 if part>=30 else 0x11220000+part) if legacy else {15:{2:0x56494431,3:0x53525631,4:0x484c5431,5:0x50415431}[phase],12:1}.get(part,part+100)
  tag=(0x4c4f0000|part) if legacy else (0x56530000|part<<3|phase)
  seq=(sequence+6*(part//2)+part%2)&255 if legacy else sequence
  packet=word|(tag<<32)|(seq<<64)|(version<<72)
  if not legacy:packet|=(1<<80)|(int(stale)<<82)
  return ((packet<<64)|0x4f465035|(5<<40)|(1<<32)|(61<<56))<<2
 def test_every_word_every_page(self):
  for phase,kind in ((2,'display'),(3,'process'),(4,'health'),(5,'path')):
   a=PageAssembler()
   for part in range(16):result=a.add(decode(self.raw(phase,part),self.positions,[]))
   self.assertEqual(result['page_kind'],kind);self.assertFalse(result['whole_page_atomic']);self.assertEqual(len(result['chunks']),16)
 def test_legacy_reconstruction_and_sequence_wrap(self):
  for seq in (0,200,247,255):
   a=PageAssembler([11,22])
   with patch('read_video_service_pages.legacy_decode',return_value={}) as old:
    for i in range(32):result=a.add(decode(self.raw(0,i,legacy=True,sequence=seq),self.positions,[]))
   self.assertEqual(result['page_kind'],'legacy');self.assertFalse(result['whole_page_atomic'])
   raw,slots=old.call_args.args;self.assertEqual(slots,[11,22]);self.assertEqual(raw>>66,sum((0x11220000+i)<<(i*32) for i in range(30)))
 def test_missing_word_rejected(self):
  a=PageAssembler()
  for i in range(32):
   if i!=27:self.assertIsNone(a.add(decode(self.raw(0,i,legacy=True),self.positions,[])))
 def test_mixed_camera_epoch_rejected(self):
  for bad in (1,24,26,29,31):
   a=PageAssembler()
   with patch('read_video_service_pages.legacy_decode') as old:
    for i in range(32):
     c=decode(self.raw(0,i,legacy=True),self.positions,[])
     if i==bad:c['page_sequence']=(c['page_sequence']+96)&255
     result=a.add(c)
    self.assertIsNone(result);old.assert_not_called()
 def test_padding_version_and_flags_rejected(self):
  for r in (self.raw(3,0,version=4),self.raw(3,0)|(1<<800),self.raw(0,31,legacy=True,word=1),self.raw(0,0,legacy=True)|(1<<(66+80))):
   with self.assertRaises(ValueError):decode(r,self.positions,[])
 def test_32bit_prepared_word_controls(self):
  for word in (0,1,0x1fff):
   c=decode(self.raw(3,12,word=word),self.positions,[])
   with patch.object(o,'fresh',return_value=c):self.assertEqual(o.prepared(None,self.positions,[],[],lambda:None),bool(word&1))
 def test_reserved_prepared_word_rejected(self):
  c=decode(self.raw(3,12,word=1<<13),self.positions,[])
  with patch.object(o,'fresh',return_value=c):
   with self.assertRaises(ValueError):o.prepared(None,self.positions,[],[],lambda:None)
 def test_stale_display_does_not_hide_process(self):
  self.assertFalse(decode(self.raw(2,0,stale=True),self.positions,[])['measurement_valid'])
  self.assertTrue(decode(self.raw(3,0,stale=True),self.positions,[])['measurement_valid'])
 def test_path_word_bounds_and_all_modes(self):
  for value in range(64):
   row=decode(self.raw(5,14,word=value),self.positions,[]);status=o.path_status(row)
   self.assertEqual(status['mode'],value&3);self.assertEqual(status['inspect_bank'],value>>4)
  self.assertIsNone(o.path_status(decode(self.raw(5,13),self.positions,[])))
  with self.assertRaises(ValueError):o.path_status(decode(self.raw(5,14,word=64),self.positions,[]))
if __name__=='__main__':unittest.main()
