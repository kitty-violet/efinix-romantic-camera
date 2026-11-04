"""No USB: bounded run command, prior-image continuity and terminal checks."""
import copy
import unittest
from unittest.mock import patch

import accepted_a_first_frame as h
from test_ddr_scratch_bringup import Link


class RunLink(Link):
    def __init__(self, result='pass'):
        super().__init__(); self.result=result
    def scan(self, value, bits):
        result=super().scan(value,bits)
        if bits==77 and self.calls[-1][0] and self.calls[-1][1]==h.host.MB:
            m=self.mem
            m[h.host.MB]=0
            m[h.host.MB+4]=5
            m[h.host.MB+28]=m[h.host.MB+12]
            m[h.host.APB+4]=(2560<<8)|5
            m[h.host.APB+0x50]=m[h.host.MB+16]
            m[h.host.APB+0x54]=m[h.host.MB+20]
            m[h.host.APB+0x58]=8105408
            m[h.host.APB+0x5c]=1
            m[h.host.APB+0x64]=h.TAG
            if self.result=='crc':m[h.host.MB+28]^=1
            if self.result=='runaway':m[h.host.APB+0x5c]=2
            if self.result=='not_stopped':m[h.host.APB+8]=3
            if self.result=='resident':m[h.host.APB+4]&=~1
            if self.result=='fault':m[h.host.MB+4]=0x80000007
        return result


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.payload=h.stage.payload_checked()

    def test_fixed_parameters_single_command_and_stop(self):
        link=RunLink();bus=h.RunBus(h.StableReplyTransport(link),self.payload);out={}
        h.run_once(bus,self.payload,out,lambda:None)
        self.assertEqual(bus.write_attempts,h.plan(self.payload))
        self.assertEqual(out['committed_frames'],1)
        self.assertTrue(out['resident_verified_by_firmware'])
        self.assertAlmostEqual(out['cnn_ms_at_150mhz'],54.03605333333333)
        with self.assertRaises(ValueError):bus.write(h.host.MB,0x434e4e31)

    def test_block_all_early_wrong_or_ddr_writes(self):
        link=RunLink();bus=h.RunBus(link,self.payload)
        for a,v in h.plan(self.payload):
            with self.assertRaises(ValueError):bus.write(a,v)
        bus.armed=True
        for a,v in ((h.host.MB,0x434e4e31),(h.host.MB+8,0),(h.host.APB+8,3),
                    (h.host.SCRATCH,1),(0xe1200000,1)):
            with self.assertRaises(ValueError):bus.write(a,v)
        self.assertEqual(link.calls,[])

    def test_parameter_mismatch_never_starts(self):
        link=RunLink()
        original=link.scan
        def scan(v,n):
            r=original(v,n)
            if n==77 and link.calls[-1][:2]==(0,h.host.MB+8):link.reply^=1
            return r
        link.scan=scan;bus=h.RunBus(h.StableReplyTransport(link),self.payload)
        with self.assertRaises(IOError):h.run_once(bus,self.payload,{},lambda:None)
        self.assertEqual(bus.write_attempts,[h.plan(self.payload)[0]])

    def test_terminal_errors_never_pass(self):
        for failure in ('crc','runaway','not_stopped','resident','fault'):
            bus=h.RunBus(h.StableReplyTransport(RunLink(failure)),self.payload)
            with self.assertRaises(RuntimeError):h.run_once(bus,self.payload,{},lambda:None)
            self.assertEqual(len(bus.write_attempts),6)

    def test_reply_error_no_command_replay(self):
        for failure in ('error','timeout'):
            link=Link(failure);bus=h.RunBus(h.StableReplyTransport(link),self.payload);bus.armed=True
            with self.assertRaises((IOError,TimeoutError)):bus.write(*bus.plan[0])
            with self.assertRaises(ValueError):bus.write(*bus.plan[0])
            self.assertEqual(len(link.calls),1)

    def test_reset_or_intervening_command_rejected(self):
        previous=h.load(h.PRIOR)['snapshots'][-1]
        h.continuity(previous,previous)
        for section,key in (('ddr','sampled_system_reset_events'),('ddr','sampled_ddr_lock_drops'),
                            ('live','accepted_commands'),('live','boot_state')):
            now=copy.deepcopy(previous);now[section][key]+=1
            with self.assertRaises(RuntimeError):h.continuity(now,previous)

    def test_retention_checks_each_page_without_writes(self):
        link=Link()
        for name,data in self.payload.items():
            base=h.host.WINDOWS[name][0]
            link.mem.update({base+off:int.from_bytes(data[off:off+4],'little') for off in range(0,len(data),4)})
        bus=h.RunBus(link,self.payload);rows=h.retention(bus,self.payload)
        self.assertEqual(sum(x['checked_words'] for x in rows),187)
        self.assertFalse(any(w for w,a,v in link.calls))
        link.mem[0xe1200000]^=1
        with self.assertRaises(IOError):h.retention(bus,self.payload)

    def test_host_poll_timeout_does_not_repeat_start(self):
        class Bus:
            plan=h.plan(self.payload)
            armed=False
            writes=[]
            def write(self,a,v):self.writes.append((a,v))
            def read(self,a):return dict(self.plan)[a]
        sample=dict(phase=4,apb={'04':0},observed_coefficient_crc=0)
        with patch.object(h.host,'idle',return_value=dict(apb={'5c':0})), \
             patch.object(h.host,'snapshot',return_value=sample), \
             patch.object(h.time,'monotonic',side_effect=[0,0,0,16]), \
             patch.object(h.time,'sleep'):
            bus=Bus()
            with self.assertRaises(TimeoutError):h.run_once(bus,self.payload,{},lambda:None)
        self.assertEqual(bus.writes,h.plan(self.payload))


if __name__=='__main__':unittest.main(verbosity=2)
