import unittest,zlib
from video_input_isp_control import table,upload_plan,check_word,verified
from video_service_board import ControlBus
from fixed_jtag_loader import MB

class ISPControlTests(unittest.TestCase):
 def test_identity_and_protocol(self):
  b=table(gamma=1);self.assertEqual(b,bytes(range(256))*3)
  p=upload_plan(b,False);self.assertEqual(len(p['words']),771)
  self.assertEqual(p['words'][1:769],[0x49000000|i<<8|(i&255) for i in range(768)])
  self.assertEqual(p['crc32'],f'{zlib.crc32(b):08x}');self.assertEqual(p['readback_rgb'][17],[17]*3)
 def test_table_finite_monotone(self):
  b=table()
  for c in range(3):
   a=b[c*256:(c+1)*256];self.assertEqual(a[0],0);self.assertEqual(a[-1],255);self.assertEqual(list(a),sorted(a))
  for kw in [dict(gamma=float('nan')),dict(gains=[1,float('inf'),1]),dict(black=-1)]:
   with self.assertRaises(ValueError):table(**kw)
 def test_reserved_and_out_of_table_rejected(self):
  for w in [0x49030000,0x49400002,0x49810000,0x49c10000,0x4a000100,0x56530000,-1,True]:
   with self.assertRaises(ValueError):check_word(w)
 def test_all_bytes_required(self):
  for b in [b'',bytes(767),bytes(769)]:
   with self.assertRaises(ValueError):upload_plan(b)
 def test_verified_actual_readback_required(self):
  p=upload_plan(table());s=dict(valid=True,updating=False,read_pending=False,read_request=False,count=768,accepted_crc32=p['crc32'],swap_rb=True,rejected=3)
  self.assertTrue(verified(p,s,p['readback_rgb'],3))
  for field,value in [('count',767),('valid',False),('accepted_crc32','00000000'),('rejected',4),('read_request',True)]:
   with self.assertRaises(ValueError):verified(p,dict(s,**{field:value}),p['readback_rgb'],3)
  r=[x[:] for x in p['readback_rgb']];r[100][1]^=1
  with self.assertRaises(ValueError):verified(p,s,r,3)
 def test_bus_remains_opt_in(self):
  b=ControlBus(None,writing=True,diagnostic=True)
  with self.assertRaises(ValueError):b.write(MB+0x28,0x49400000)
  b=ControlBus(None,writing=True,input_isp=True)
  with self.assertRaises(ValueError):b.write(MB+0x28,0x56530000)
  self.assertEqual(b.attempts,[])
if __name__=='__main__':unittest.main()
