import unittest
from unittest.mock import patch
import video_service_board as b

class Link:
 def __init__(self,reply=(0x54493630<<4)|4):self.calls=[];self.reply=reply
 def scan(self,v,n):self.calls.append((v,n));return self.reply if n==36 else 0

class BoardTests(unittest.TestCase):
 def test_readonly_cannot_write_or_access_ddr(self):
  t=Link();bus=b.ControlBus(t)
  for writing,address in ((True,b.MB),(False,0xe1000000),(False,b.MB+0x28),(False,b.APB+1)):
   with self.assertRaises(ValueError):bus.access(writing,address)
  self.assertEqual(t.calls,[])
 def test_direct_status_uses_one_command(self):
  t=Link();bus=b.ControlBus(t);self.assertEqual(bus.read(b.APB),0x54493630)
  self.assertEqual([n for v,n in t.calls],[77,36]);self.assertEqual(bus.transactions,1)
 def test_write_timeout_is_not_replayed(self):
  t=Link(0);bus=b.ControlBus(t,writing=True)
  with self.assertRaises(TimeoutError):bus.write(b.MB,b.LIVE)
  self.assertEqual(sum(n==77 for v,n in t.calls),1);self.assertEqual(bus.errors,1)
 def test_reply_error_stops_without_replay(self):
  t=Link(12);bus=b.ControlBus(t,writing=True)
  with self.assertRaises(IOError):bus.write(b.MB,b.STOP)
  self.assertEqual(len(t.calls),2);self.assertEqual(bus.errors,1)
 def test_control_bus_excludes_image_and_diagnostic_writes(self):
  t=Link();bus=b.ControlBus(t,writing=True)
  for address in (0xe1000000,b.APB,b.MB+4,b.MB+0x28):
   with self.assertRaises(ValueError):bus.write(address,0)
  self.assertEqual(t.calls,[])
 def test_expired_session_sends_no_packet(self):
  t=Link();bus=b.ControlBus(t,writing=True,seconds=0)
  with self.assertRaises(TimeoutError):bus.write(b.MB,b.LOAD)
  self.assertEqual(t.calls,[])
 def test_old_release_rejected_before_usb(self):
  class Fake:
   def read_text(self,**k):return '{"status":"OLD_RELEASE"}'
  with self.assertRaises(ValueError):b.bound(Fake())
 def test_diagnostic_word_whitelist(self):
  t=Link();bus=b.ControlBus(t,writing=True,diagnostic=True)
  for bad in (0,0x56530002,0x56530004,0x56530800,0x56531400,0x56530000+(1<<32)):
   with self.assertRaises(ValueError):bus.write(b.MB+0x28,bad)
  self.assertEqual(t.calls,[])
  bus.write(b.MB+0x28,b.diagnostic_word(tx_pattern=True,mode=2))
  self.assertEqual(sum(n==77 for v,n in t.calls),1)
 def test_diagnostic_capability_cannot_write_ram(self):
  t=Link();bus=b.ControlBus(t,writing=True,diagnostic=True,read_window=True)
  with self.assertRaises(ValueError):bus.write(0xe1000000,1)
  with self.assertRaises(ValueError):bus.read(0xe1400000)
  self.assertEqual(t.calls,[])
  self.assertEqual(bus.read(0xe1000000),0x54493630)
 def test_window_bounds_checked_before_access(self):
  t=Link();bus=b.ControlBus(t,read_window=True)
  for offset,size in [(-4,4),(0,4097),(0,0),(1,4),(0x400000,4),(0x3ffffc,8)]:
   with self.assertRaises(ValueError):b.read_window(bus,offset,size)
  self.assertEqual(t.calls,[])
  self.assertEqual(b.read_window(bus,0x3ffffc,4),bytes.fromhex('30364954'))
 def test_pattern_diagnostic_does_not_require_healthy_pipeline(self):
  class Registers:
   def __init__(self):self.word=1;self.writes=[]
   def read(self,address):return self.word
   def write(self,address,value):self.writes.append((address,value));self.word=value
  bus=Registers();report={};path=dict(mode=0,inspect_enabled=False,inspect_result=False,inspect_bank=0)
  with patch.object(b,'snapshot',side_effect=AssertionError('Pattern independent of pipeline status')):
   self.assertEqual(b.diagnostic_once(bus,report,lambda:None,read_path=lambda:path,clear_first=True),path)
  self.assertEqual(bus.writes,[(b.MB+0x28,0x56530101)])
 def test_rejected_mode_detected_after_one_write(self):
  class Registers:
   def __init__(self):self.word=0;self.writes=[]
   def read(self,address):return 0 if address==b.MB else self.word
   def write(self,address,value):self.writes.append((address,value));self.word=value
  bus=Registers()
  with patch.object(b,'snapshot',return_value={'phase':1,'apb':{'08':0}}):
   with self.assertRaises(RuntimeError):b.diagnostic_once(bus,{},lambda:None,read_path=lambda:dict(mode=0),mode=2)
  self.assertEqual(len(bus.writes),1)
 def test_running_mode_change_is_not_submitted(self):
  bus=Link()
  with patch.object(b,'snapshot',return_value={'phase':4,'apb':{'08':3}}):
   with self.assertRaises(RuntimeError):b.diagnostic_once(bus,{},lambda:None,read_path=lambda:{},mode=1)
  self.assertEqual(bus.calls,[])
if __name__=='__main__':unittest.main()
