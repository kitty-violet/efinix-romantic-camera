import unittest
from stable_jtag_reply import StableReplyTransport
from fixed_jtag_loader import Bus,APB,decode_response

def reply(data,valid=1,error=0):return ((data<<2)|(error<<1)|valid)<<2

class Mock:
    def __init__(self,replies):self.replies=iter(replies);self.calls=[]
    def scan(self,v,n):
        self.calls.append((v,n))
        return next(self.replies) if (v,n)==(1,36) else 0
    def close(self):pass

class Tests(unittest.TestCase):
    def test_wait_for_stable_no_repeat_command(self):
        r=reply(0x54493630);m=Mock([0,r,r+16,r,r,r]);s=StableReplyTransport(m)
        self.assertEqual(Bus(s).read(APB),0x54493630)
        self.assertEqual(sum(n==77 for v,n in m.calls),1)
        self.assertEqual(s.poll_pairs,3);self.assertEqual(s.unstable_pairs,2)
    def test_stable_error_propagates(self):
        r=reply(0,error=1);s=StableReplyTransport(Mock([r,r]))
        with self.assertRaises(IOError):Bus(s).read(APB)
    def test_ignore_wrapper_bits(self):
        r=reply(123);self.assertEqual(decode_response(StableReplyTransport(Mock([r|1,r|2])).scan(1,36)),123)
    def test_preserve_observation_scan(self):
        m=Mock([]);StableReplyTransport(m).scan(3,386);self.assertEqual(m.calls,[(3,386)])
    def test_address_policy_unchanged(self):
        m=Mock([])
        with self.assertRaises(ValueError):Bus(StableReplyTransport(m)).write(0,1)
        self.assertEqual(m.calls,[])
    def test_invalid_never_commits(self):
        m=Mock([0]*400);b=Bus(StableReplyTransport(m))
        with self.assertRaises(TimeoutError):b.read(APB)
        self.assertEqual(b.transactions,0);self.assertEqual(b.errors,1)
        self.assertEqual(sum(n==77 for v,n in m.calls),1)

if __name__=='__main__':unittest.main()
