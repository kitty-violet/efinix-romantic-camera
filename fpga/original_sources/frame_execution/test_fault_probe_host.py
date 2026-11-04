import re,unittest
from unittest.mock import patch
from run_units import ROOT
from read_fault_probe import decode,frequency_report,MAGIC,SCAN_BITS

class FaultProbeHostTests(unittest.TestCase):
 def packet(self,adapter=0,clock=0):
  value=MAGIC|(1<<32)|(4<<40)|(adapter<<448)|(clock<<576)
  return value<<2
 def test_lengths_and_versions(self):
  for value in (-1,True,1<<SCAN_BITS,self.packet()^(1<<2),self.packet()^(1<<42)):
   with self.assertRaises(ValueError):decode(value)
 def test_adapter_fields(self):
  a=(0x8123fedc<<96)|(1<<95)|(3<<88)|(0x135724<<64)|(0xe1001234<<32)|0x12ab9876
  r=decode(self.packet(a))['adapter']
  self.assertTrue(r['captured']);self.assertEqual(r['captured_error_mask'],3)
  self.assertEqual(r['tick'],0x135724);self.assertEqual(r['address'],'e1001234')
  self.assertEqual(r['preceding_cycle']['raw'],'12ab9876');self.assertEqual(r['live']['raw'],'8123fedc')
 def test_clock_fields(self):
  c=sum((i+100)<<(24*i) for i in range(5))|(0xf5<<120)|(0x1122<<128)|(0x3344<<144)|(0x5566<<160)|(7<<176)|(8<<184)
  r=decode(self.packet(clock=c))['clock_probe']
  self.assertEqual(list(r['counts'].values()),[100,101,102,103]);self.assertEqual(r['ticks'],104)
  self.assertEqual((r['reset_age'],r['last_low'],r['last_high'],r['reset_rises'],r['reset_falls']),(0x1122,0x3344,0x5566,7,8))
 def test_actual_rtl_serial_bytes(self):
  log=ROOT/'tmp/frame-execution/fault-probe-short2/serial.log'
  raw=[int(x,16) for x in re.findall(r'OFP4RAW ([0-9a-fA-F]+)',log.read_text())]
  self.assertEqual(len(raw),2)
  expected=sum((0x913457a0+i)<<(32*i) for i in range(22))
  for n,v in enumerate(raw):
   payload=expected if n==0 else expected^((1<<704)-1)
   self.assertEqual(v>>64,payload)
   # Walking data intentionally violates legacy reserved control fields.
   # Keep the exact new extension bytes, and provide a valid zero legacy
   # payload to its strict decoder; the full transport was compared above.
   compatible=v&~(((1<<384)-1)<<64)
   decoded=decode(compatible<<2)
   self.assertEqual(decoded['adapter']['live']['raw'],f'{payload>>480&0xffffffff:08x}')
   self.assertEqual(decoded['clock_probe']['counts']['core200'],payload>>512&0xffffff)
 def test_frequency_ratio_and_wrap(self):
  def sample(t,c):return dict(clock_probe=dict(ticks=t,counts=dict(core200=c,tdqss400=2*c,tac400=2*c,twd400=2*c)))
  a=sample((1<<24)-300,(1<<24)-400);b=sample(300,400)
  # Independently wrap each source counter as the hardware does.
  for record in (a,b):
   record['clock_probe']['counts']={n:v%(1<<24) for n,v in record['clock_probe']['counts'].items()}
  r=frequency_report([a,b]);self.assertEqual(list(r['mhz_relative_to_process150'].values()),[200,400,400,400])
  self.assertEqual(frequency_report([a,a])['status'],'INVALID_COMPARISON_WINDOW')
 def test_bounded_status_uses_ofp4(self):
  import read_fault_probe_status as reader
  from test_command_progress_status import Transport,Tests
  self.assertEqual(reader.fresh.__module__,'read_fault_probe')
  out=Tests().output()
  with patch.object(reader,'fresh',side_effect=Tests().snapshots()):reader.inspect(Transport(),out,lambda:None)
  self.assertEqual(out['status'],'OFP4_AND_STABLE_STATUS_READ_COMPLETE_NOT_BOARD_PASS')
  self.assertEqual(out['transactions'],15)
 def test_timeout_keeps_independent_ofp4_evidence(self):
  import read_fault_probe_status as reader
  from test_command_progress_status import Transport,Tests
  out=Tests().output()
  with patch.object(reader,'fresh',return_value={'levels':{'control_reset':False},'format':'OFP4'}):
   with self.assertRaises(TimeoutError):reader.inspect(Transport(True),out,lambda:None)
  self.assertEqual(len(out['snapshots']),2)

if __name__=='__main__':unittest.main(verbosity=2)
