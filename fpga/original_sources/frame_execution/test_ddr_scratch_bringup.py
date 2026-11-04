"""No hardware: reject out-of-scope writes, corruption, stalled/error responses."""
import unittest
import ddr_scratch_bringup as h

class Link:
    def __init__(self, fault=None):
        self.mem={h.APB:0x54493630,h.APB+12:0xb1000000,h.APB+0x6c:0x105b43ff,h.MB+4:1}
        self.calls=[];self.reply=0;self.fault=fault
    def scan(self,v,n):
        if n==77:
            x=v>>2;address=(x>>8)&0xffffffff;wr=(x>>72)&1;data=(x>>40)&0xffffffff
            self.calls.append((wr,address,data));self.reply=self.mem.get(address,0)
            if wr:self.mem[address]=data
            if self.fault=='corrupt' and not wr and address==h.SCRATCH:self.reply^=1
            return 0
        assert (v,n)==(1,36)
        if self.fault=='timeout':return 0
        return ((self.reply<<2)|1|(2 if self.fault=='error' else 0))<<2

class Checks(unittest.TestCase):
    def test_whole_scratch_and_guards(self):
        link=Link();bus=h.ScratchBus(h.StableReplyTransport(link))
        out=dict(patterns=[],guards=[]);h.exercise(bus,out,lambda:None)
        self.assertEqual([r['status'] for r in out['patterns']],['PASS']*3)
        self.assertEqual(bus.writes,2052);self.assertEqual(bus.transactions,4132)
        self.assertTrue(all(h.SCRATCH<=a<h.SCRATCH+4096 for w,a,v in link.calls if w))
        self.assertTrue(all(r['actual_crc32']==r['expected_crc32'] for r in out['patterns']))
    def test_reject_outside_and_unaligned_before_any_scan(self):
        link=Link();bus=h.ScratchBus(link)
        for a in (h.APB,h.MB,0xe1000000,0xe1100000,0xe1200000,h.SCRATCH-4,h.SCRATCH+4096,h.SCRATCH+1):
            with self.assertRaises(ValueError):bus.write(a,0)
        self.assertEqual(link.calls,[])
    def test_error_timeout_no_write_retry(self):
        for fault in ('error','timeout'):
            link=Link(fault);bus=h.ScratchBus(h.StableReplyTransport(link))
            with self.assertRaises((IOError,TimeoutError)):bus.write(h.SCRATCH,1)
            self.assertEqual(len(link.calls),1);self.assertEqual(bus.errors,1)
    def test_corruption_saves_first_mismatch(self):
        link=Link('corrupt');bus=h.ScratchBus(h.StableReplyTransport(link));out=dict(patterns=[],guards=[])
        with self.assertRaises(IOError):h.exercise(bus,out,lambda:None)
        self.assertEqual(len(out['patterns']),1);self.assertEqual(out['patterns'][0]['mismatch_count'],1)
        self.assertEqual(bus.writes,4)
    def test_bad_calibration_prevents_all_writes(self):
        link=Link();link.mem[h.APB+0x6c]=0
        with self.assertRaises(RuntimeError):h.exercise(h.ScratchBus(link),dict(patterns=[],guards=[]),lambda:None)
        self.assertFalse(any(w for w,a,v in link.calls))
    def test_loaded_or_running_refused(self):
        for a,v in ((h.APB+4,1),(h.APB+8,1),(h.MB,0x434e4e31),(h.MB+4,2)):
            link=Link();link.mem[a]=v
            with self.assertRaises(RuntimeError):h.guard(h.ScratchBus(link))
    def test_total_deadline_and_command_budget(self):
        link=Link();bus=h.ScratchBus(link);bus.deadline=0
        with self.assertRaises(TimeoutError):bus.write(h.SCRATCH,0)
        bus=h.ScratchBus(link);bus.attempted_commands=5000
        with self.assertRaises(RuntimeError):bus.write(h.SCRATCH,0)
        self.assertEqual(link.calls,[])

if __name__=='__main__':unittest.main(verbosity=2)
