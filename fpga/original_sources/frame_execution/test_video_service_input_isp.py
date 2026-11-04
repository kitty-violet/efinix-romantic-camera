import unittest
from video_input_isp_control import table,upload_plan
from video_service_input_isp import configure,decode_status,MB

class Tests(unittest.TestCase):
 def setup_case(self):
  plan=upload_plan(table());report={};writes=[]
  class Bus:
   def write(self,address,value):
    if address!=MB+0x28:raise AssertionError('out of scope address')
    writes.append(value)
   def read(self,address):return writes[-1]
  def status(index):
   return dict(valid=True,updating=False,count=768,accepted_crc32=plan['crc32'],swap_rb=True,
    read_pending=False,read_request=False,rejected=2,read_index=index or 0,read_rgb=plan['readback_rgb'][index or 0])
  return plan,report,writes,Bus(),status
 def test_upload_once_complete_readback_and_stop(self):
  plan,r,w,b,status=self.setup_case();guards=[]
  configure(b,plan,r,lambda:None,check_stopped=lambda:guards.append(1),read_status=status)
  self.assertEqual(w,plan['words']+[0x4a000000+i for i in range(256)])
  self.assertEqual(r['readbacks'],plan['readback_rgb']);self.assertGreater(len(guards),20)
  self.assertEqual(r['status'],'INPUT_ISP_CRC_AND_ALL_768_BYTES_VERIFIED_STOPPED')
 def test_lost_write_reply_never_replayed(self):
  p,r,w,b,s=self.setup_case()
  def fail(address):raise TimeoutError('lost reply')
  b.read=fail
  with self.assertRaises(TimeoutError):configure(b,p,r,lambda:None,check_stopped=lambda:None,read_status=s)
  self.assertEqual(w,p['words'][:1])
 def test_active_run_cannot_write(self):
  p,r,w,b,s=self.setup_case()
  def active():raise ValueError('RUN')
  with self.assertRaises(ValueError):configure(b,p,r,lambda:None,check_stopped=active,read_status=s)
  self.assertEqual(w,[])
 def test_bad_ram_readback_fails(self):
  p,r,w,b,s=self.setup_case()
  def corrupt(index):
   x=s(index)
   if index==89:x['read_rgb']=[0,0,0]
   return x
  with self.assertRaises(ValueError):configure(b,p,r,lambda:None,check_stopped=lambda:None,read_status=corrupt)
  self.assertNotEqual(r.get('status'),'INPUT_ISP_CRC_AND_ALL_768_BYTES_VERIFIED_STOPPED')
 def test_rejected_commit_no_reads(self):
  p,r,w,b,s=self.setup_case()
  def reject(index):
   x=s(index)
   if w:x['valid']=False
   return x
  with self.assertRaises(ValueError):configure(b,p,r,lambda:None,check_stopped=lambda:None,read_status=reject)
  self.assertEqual(w,p['words'])
 def test_word_order_and_reserved(self):
  x=decode_status([0x12345678,(2<<16)|768|1024|4096,0x09ccbbaa,0xe483a10c])
  self.assertEqual(x['read_rgb'],[0xaa,0xbb,0xcc]);self.assertEqual(x['read_index'],9)
  self.assertTrue(x['valid']);self.assertTrue(x['swap_rb']);self.assertEqual(x['count'],768)
  self.assertEqual(x['accepted_crc32'],'e483a10c');self.assertEqual(x['rejected'],2)
  self.assertRaises(ValueError,decode_status,[0,0x8000,0,0])
if __name__=='__main__':unittest.main()
