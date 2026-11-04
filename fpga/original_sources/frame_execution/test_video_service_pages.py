import json,unittest
from pathlib import Path
from read_video_service_pages import decode,PageAssembler
ROOT=Path(__file__).resolve().parents[4]
class PageTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.positions=json.loads((ROOT/'tmp/frame-execution/video-service-native10/graft.json').read_text())['display_extension']['observation_positions']
 def raw(self,phase=2,stale=False,valid=True):
  words=[100+i for i in range(16)];words[15]={3:0x53525631,5:0x484c5431,7:0x50415431}.get(phase,0x56494431)
  packet=sum(w<<(32*i) for i,w in enumerate(words)) | ((0x56530000|phase)<<512) | (87<<544) | (1<<552) | (int(valid)<<560) | (int(stale)<<562)
  payload=sum(((packet>>i)&1)<<p for i,p in enumerate(self.positions))
  return ((payload<<64)|0x4f465035|(5<<40)|(1<<32)|(61<<56))<<2
 def test_display_scatter(self):
  d=decode(self.raw(),self.positions,[]);self.assertEqual(d['display']['complete_frames'],112);self.assertEqual(d['page_sequence'],87)
 def test_stale_retains_data_but_invalid_measurement(self):
  d=decode(self.raw(stale=True),self.positions,[]);self.assertEqual(d['display']['physical_frames'],113);self.assertFalse(d['display']['measurement_valid'])
 def test_no_clock_no_fake_zero_measurement(self):
  self.assertIsNone(decode(self.raw(valid=False),self.positions,[])['display'])
 def test_process_counters(self):
  d=decode(self.raw(3),self.positions,[]);self.assertEqual(d['process']['committed'],107);self.assertEqual(d['process']['pixel_cache_timeouts'],103)
 def test_bad_signature(self):
  with self.assertRaises(ValueError):decode(self.raw()^4,self.positions,[])
 def test_health_independent_of_pixel_clock(self):
  d=decode(self.raw(5,stale=True,valid=False),self.positions,[]);self.assertEqual(d['health']['source_area_frames'],112)
 def test_path_not_mistaken_for_ai_rate(self):
  d=decode(self.raw(7),self.positions,[]);self.assertEqual(d['path']['mode'],'DDR_PATTERN');self.assertFalse(d['path']['counts_are_ai_results'])
 def test_bad_manifest(self):
  with self.assertRaises(ValueError):decode(self.raw(),self.positions[:-1]+[0],[])
class ChunkTests(unittest.TestCase):
 positions=list(range(0,366,2))
 def raw(self,phase,part,stale=False,pixel_valid=True):
  words=[100+i for i in range(16)];words[15]={2:0x56494431,3:0x53525631,4:0x484c5431,5:0x50415431}[phase]
  data=sum(w<<(32*i) for i,w in enumerate(words[part*4:part*4+4]))
  packet=data|((0x56530000|part<<3|phase)<<128)|((part*6+phase)<<160)|(2<<168)|(int(pixel_valid)<<176)|(int(stale)<<178)
  payload=sum(((packet>>i)&1)<<p for i,p in enumerate(self.positions))
  return ((payload<<64)|0x4f465035|(5<<40)|(1<<32)|(61<<56))<<2
 def test_all_pages_chunks_and_separate_time(self):
  for phase,kind in ((2,'display'),(3,'process'),(4,'health'),(5,'path')):
   assembler=PageAssembler()
   for part in range(4):
    c=decode(self.raw(phase,part),self.positions,[])
    self.assertEqual(c['chunk_index'],part);self.assertFalse(c['whole_page_atomic'])
    result=assembler.add(c)
    if part!=3:self.assertIsNone(result)
   self.assertEqual(result['page_kind'],kind);self.assertFalse(result['whole_page_atomic']);self.assertEqual(len(result['chunks']),4)
 def test_missing_or_repeated_chunk_not_assembled(self):
  for order in ((0,2,3),(0,1,1,2,3),(1,2,3)):
   a=PageAssembler()
   for part in order:self.assertIsNone(a.add(decode(self.raw(3,part),self.positions,[])))
 def test_stale_pixel_invalidates_partial(self):
  a=PageAssembler();a.add(decode(self.raw(2,0),self.positions,[]))
  self.assertIsNone(a.add(decode(self.raw(2,1,stale=True),self.positions,[])))
  self.assertIsNone(a.add(decode(self.raw(2,2),self.positions,[])))
 def test_dead_pixel_does_not_block_health(self):
  a=PageAssembler()
  for part in range(4):r=a.add(decode(self.raw(4,part,stale=True,pixel_valid=False),self.positions,[]))
  self.assertEqual(r['fields']['source_area_frames'],112)
 def test_chunk_manifest_duplicates_rejected(self):
  with self.assertRaises(ValueError):decode(self.raw(2,0),self.positions[:-1]+[0],[])
if __name__=='__main__':unittest.main()
