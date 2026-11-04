import unittest,re
import read_first_fault as f

def packet(seq=0,valid=1,busy=0,req=0,ack=0,words=None):
    value=f.MAGIC|valid<<32|busy<<33|ack<<34|req<<35|1<<40|0x3f<<48|seq<<56
    for i,w in enumerate(words or [0]*6):value|=w<<(64+32*i)
    return value<<2

class Codec(unittest.TestCase):
    def test_state_labels_match_frozen_rtl(self):
        from first_fault_observer import BASE
        text=(BASE/'assembly/EfxSapphireSoc.v').read_text()
        line=next(s for s in text.splitlines() if 'localparam [9:0] IDLE=' in s)
        for name,value in re.findall(r"(\w+)=10'h([0-9a-f]+)",line):
            self.assertEqual(f.state(int(value,16))['asserted'],[name])
    def test_actual_fields(self):
        meta=3<<30|1<<26|3<<20|2<<10|0
        d=f.decode(packet(words=[5,meta,0xf8100040,0x00800000,0x00010000,0xf9000f00]))['payload']
        self.assertTrue(d['captured']);self.assertTrue(d['armed']);self.assertEqual(d['boot_state'],3)
        self.assertFalse(d['control_state']['legal']);self.assertEqual(d['previous_control_state']['asserted'],['DECODE'])
        self.assertEqual(d['flags']['boot_fault'],32);self.assertEqual(d['address'],'f8100040')
        self.assertTrue(d['reason']['illegal_control_state']);self.assertEqual(d['previous_address'],'f9000f00')
    def test_reject_old_signature(self):
        with self.assertRaises(ValueError):f.decode(packet()^((f.MAGIC^0x4f425331)<<2))
    def test_reject_bad_header(self):
        for delta in (1<<36,2<<40):
            with self.assertRaises(ValueError):f.decode(packet()^(delta<<2))
    def test_reject_bad_handshake(self):
        with self.assertRaises(ValueError):f.decode(packet(busy=1))
    def test_not_captured_is_distinct(self):
        d=f.decode(packet(words=[0,1<<30,0,0,0,0]))['payload']
        self.assertTrue(d['armed']);self.assertFalse(d['captured'])
        self.assertIsNone(f.decode(packet(valid=0))['payload'])
    def test_illegal_two_hot(self):self.assertFalse(f.state(3)['legal'])
    def test_fresh_wrap_and_selectors(self):
        replies=iter([packet(255),packet(255,busy=1,req=1),packet(0)])
        calls=[]
        class Transport:
            def scan(self,selector,bits):calls.append((selector,bits));return next(replies)
        history=[];out=f.fresh(Transport(),history,lambda:None)
        self.assertEqual(out['sequence'],0);self.assertEqual(calls,[(3,258),(2,258),(3,258)])
    def test_no_reissue_while_pending(self):
        calls=[]
        class Transport:
            def scan(self,selector,bits):calls.append(selector);return packet(busy=1,req=1)
        with self.assertRaises(f.SnapshotPending):f.fresh(Transport(),[],lambda:None)
        self.assertEqual(calls,[3]*9)
if __name__=='__main__':unittest.main()
