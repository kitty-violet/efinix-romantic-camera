"""Decode real RTL serial receipts, reject wrong versions, enforce snapshot protocol."""
import re,unittest
from run_units import ROOT
from read_control_progress import decode,fresh,SCAN_BITS
from read_control_snapshot import SnapshotPending

class ProgressHost(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  text=(ROOT/'tmp/frame-execution/command-progress-short2/simulation.log').read_text()
  cls.raw=int(re.findall('PROGRESSRAW ([0-9a-fA-F]+)',text)[0],16)
 def test_actual_rtl_words(self):
  r=decode(self.raw)
  self.assertEqual(r['live']['ticks'],0x89abcdef)
  self.assertEqual(r['live']['accepted_commands'],0x34)
  self.assertEqual(r['live']['received_last'],0x56)
  self.assertEqual(r['payload']['address'],'f8100000')
 def test_wrong_version_and_magic(self):
  for raw in (self.raw^(1<<42),self.raw^4):
   with self.assertRaises(ValueError):decode(raw)
 def test_bad_sizes(self):
  for raw in (-1,True,1<<SCAN_BITS):
   with self.assertRaises(ValueError):decode(raw)
 def test_reserved_bits(self):
  with self.assertRaises(ValueError):decode(self.raw|(1<<(288+29+2)))
 def test_stale_record(self):
  source=self
  class Frozen:
   def scan(self,sel,n):
    source.assertIn(sel,(2,3));source.assertEqual(n,386);return source.raw
  with self.assertRaises(SnapshotPending):fresh(Frozen(),[],lambda:None)
 def test_sequence_wrap(self):
  source=self;history=[]
  class Wrap:
   def __init__(self):self.seq=255
   def scan(self,sel,n):
    source.assertIn(sel,(2,3));source.assertEqual(n,386)
    raw=(source.raw&~(255<<58))|(self.seq<<58)
    if sel==2:self.seq=0
    return raw
  r=fresh(Wrap(),history,lambda:None)
  self.assertEqual(r['sequence'],0)
  self.assertEqual([r['selector'] for r in history],[3,2,3])
if __name__=='__main__':unittest.main()
