import unittest
from unittest.mock import patch
from read_video_service_pages import decode,PageAssembler

def raw(phase,part,word=0,version=6,legacy=False):
 tag=(0x4c4f0000|part) if legacy else (0x56530000|part<<3|phase)
 seq=(247+7*(part//2)+part%2)&255 if legacy else 31
 packet=word|(tag<<32)|(seq<<64)|(version<<72)|(0 if legacy else 1<<80)
 return ((packet<<64)|0x4f465035|(5<<40)|(1<<32)|(61<<56))<<2

class InputPages(unittest.TestCase):
 def test_new_page_fields(self):
  words=[0x12345678,0x00350400|0x1000|768,0x19563412,0xdeadbeef]+[0]*11+[0x49535031]
  a=PageAssembler()
  for i,w in enumerate(words):v=a.add(decode(raw(6,i,w),list(range(96)),[]))
  f=v['fields'];self.assertEqual(f['accepted_crc32'],'deadbeef');self.assertEqual(f['read_rgb'],[0x12,0x34,0x56])
  self.assertTrue(f['valid']);self.assertFalse(f['updating']);self.assertEqual(f['count'],768);self.assertEqual(f['rejected'],53)
 def test_legacy_epoch_new_length(self):
  a=PageAssembler()
  with patch('read_video_service_pages.legacy_decode',return_value={}) as old:
   for i in range(32):v=a.add(decode(raw(0,i,legacy=True),list(range(96)),[]))
  self.assertEqual(v['page_kind'],'legacy');old.assert_called_once()
 def test_schema5_rejects_new_phase(self):
  with self.assertRaises(ValueError):decode(raw(6,0,version=5),list(range(96)),[])
 def test_unknown_version_rejected(self):
  with self.assertRaises(ValueError):decode(raw(6,0,version=7),list(range(96)),[])
 def test_mixed_versions_never_assemble(self):
  a=PageAssembler()
  for i in range(16):
   w=0x53525631 if i==15 else 0
   v=a.add(decode(raw(3,i,w,version=5 if i==4 else 6),list(range(96)),[]))
  self.assertIsNone(v)
if __name__=='__main__':unittest.main()
