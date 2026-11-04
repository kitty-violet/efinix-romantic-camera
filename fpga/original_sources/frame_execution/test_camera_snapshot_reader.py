"""No USB imports: actual packed snapshots test sequence freshness and errors."""
import json,unittest
from run_units import ROOT
from read_camera_checked_snapshot import fresh
from read_control_snapshot import SnapshotPending

class Transport:
 def __init__(self,rows):self.rows=iter(rows);self.calls=[]
 def scan(self,selector,bits):self.calls.append((selector,bits));return next(self.rows)

class CameraReaderTests(unittest.TestCase):
 def setUp(self):
  self.slots=json.loads((ROOT/'tmp/frame-execution/connected-board-camera-checked3/camera_graft.json').read_text())['diagnostic_slots']
 def packet(self,seq,busy=False,valid=True):
  v=0x4f465035|(5<<40)|(seq<<56)|(int(valid)<<32)
  if busy:v|=(1<<33)|(1<<35)
  cfg=0xf00a3e092a2acd6b
  for i,s in enumerate(self.slots):v|=((cfg>>i)&1)<<(896+s)
  return v<<2
 def read(self,t):return fresh(t,self.slots,[],lambda:None)
 def test_fresh(self):
  t=Transport([self.packet(7),self.packet(7,busy=True),self.packet(7),self.packet(8)])
  d=self.read(t);self.assertEqual(d['sequence'],8);self.assertTrue(d['camera']['config_verified']);self.assertEqual(t.calls,[(3,1026),(2,1026),(3,1026),(3,1026)])
 def test_wrap(self):self.assertEqual(self.read(Transport([self.packet(255),self.packet(255),self.packet(0)]))['sequence'],0)
 def test_busy(self):
  t=Transport([self.packet(1,busy=True)]*9)
  with self.assertRaises(SnapshotPending):self.read(t)
  self.assertTrue(all(s==3 for s,n in t.calls))
 def test_stale(self):
  with self.assertRaises(SnapshotPending):self.read(Transport([self.packet(1)]*10))
 def test_wrong_format(self):
  with self.assertRaises(ValueError):self.read(Transport([0]))
if __name__=='__main__':unittest.main()
