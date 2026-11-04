import unittest
from read_ddr_observer import APB, OFFSETS, read_word, frequency_bounds, decode_status, healthy_snapshot


class Fake:
    def __init__(self, value=0x12345678, error=False, timeout=False):
        self.calls=[]; self.value=value; self.error=error; self.timeout=timeout
    def scan(self, packet, bits):
        self.calls.append((packet,bits))
        if bits==77:
            raw=packet>>2
            assert raw>>72&1==0
            assert (raw>>8&0xffffffff)-APB in OFFSETS
            return 0
        assert (packet,bits)==(1,36)
        return 0 if self.timeout else ((self.value<<2)|1|(2 if self.error else 0))<<2


class Tests(unittest.TestCase):
    def test_allowlist_and_no_write(self):
        for offset in OFFSETS:
            t=Fake();rows=[]
            self.assertEqual(read_word(t,offset,rows,lambda:None),t.value)
            self.assertEqual(len(t.calls),2)
            self.assertEqual(rows[0]['status'],'READ')
    def test_reject_before_transport(self):
        for offset in (True,1,0x68,0x98,0xe1000000,-4):
            t=Fake()
            with self.assertRaises(ValueError):read_word(t,offset,[],lambda:None)
            self.assertFalse(t.calls)
    def test_error_prefix(self):
        t=Fake(error=True);rows=[]
        with self.assertRaises(IOError):read_word(t,0,rows,lambda:None)
        self.assertEqual(len(t.calls),2);self.assertEqual(rows[0]['status'],'FAIL')
    def test_timeout_no_retry_command(self):
        t=Fake(timeout=True);rows=[]
        with self.assertRaises(TimeoutError):read_word(t,0,rows,lambda:None)
        self.assertEqual(sum(b==77 for _,b in t.calls),1)
        self.assertLessEqual(len(t.calls),201)
    def test_counter_wrap_and_bounds(self):
        a=dict(tick_before=0xfffffe00,tick_after=0xfffffe10,counts={n:0xffffff00 for n in ('core','tdqss','tac','twd')})
        b=dict(tick_before=0x400,tick_after=0x410,counts={'core':0x700,'tdqss':0xf00,'tac':0xf00,'twd':0xf00})
        for n,row in frequency_bounds(a,b).items():
            expected=200 if n=='core' else 400
            self.assertLess(row['mhz_lower'],expected);self.assertGreater(row['mhz_upper'],expected)
    def test_invalid_bracket(self):
        g=dict(tick_before=100,tick_after=110,counts=dict(core=0,tdqss=0,tac=0,twd=0))
        with self.assertRaises(ValueError):frequency_bounds(g,g)
    def test_status_fields(self):
        d=decode_status((255<<12)|(4<<7)|0x70|(7<<20),(3<<16)|4,(11<<16)|12,(7<<16)|8)
        self.assertEqual(d['phase_codes'],list(range(8)))
        self.assertEqual(d['selections_seen'],'04')
        self.assertEqual((d['reset_rises'],d['reset_falls'],d['system_reset_falls']),(4,3,7))
        self.assertTrue(d['age_saturated']);self.assertEqual(d['last_low_ticks'],12)
        with self.assertRaises(ValueError):decode_status(1<<23,0,0,0)
    def test_fault_guard(self):
        s=dict(levels=dict(system_reset_n=True,control_reset=False),payload=dict(armed=True,captured=False))
        healthy_snapshot(s);s['payload']['captured']=True
        with self.assertRaises(RuntimeError):healthy_snapshot(s)


if __name__=='__main__':unittest.main()
