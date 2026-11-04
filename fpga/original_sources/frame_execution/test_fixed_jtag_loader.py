"""Offline transport fault and full payload readback regressions. No USB opened."""
import copy
import hashlib
import random
import shutil
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch, Mock
import fixed_jtag_loader as host
from accepted_a_assets import ROOT, accepted_package, read_json, write_json
from network_ab.training.b1_trained_image import decode

INTEGRATION=ROOT/'tmp/frame-execution/accepted-a-integration1'


class LinkModel:
    def __init__(self, invalid=False, never=False):
        self.memory={host.APB:0x54493630,host.MB+4:1,host.APB+8:0};self.reply=None
        self.invalid=invalid;self.never=never;self.writes=[];self.scans=0

    def scan(self,value,bits):
        self.scans+=1
        if bits==77:
            assert value&3==0
            # Independently follow the public RTL command concatenation.
            w=value>>2;assert (w&255)==0 and (w>>73)==2
            a=(w>>8)&0xffffffff;v=(w>>40)&0xffffffff;wr=(w>>72)&1
            if wr:self.memory[a]=v;self.writes.append((a,v))
            if wr and a==host.MB:
                assert v==0x434e4e31
                self.memory[host.MB+4]=5
                self.memory[host.MB+28]=self.memory[host.MB+12]
                self.memory[host.APB+0x5c]=self.memory[host.MB+24]
            self.reply=self.memory.get(a,0)
            return 0
        assert bits==36 and value==1
        if self.never:return 0
        return ((self.reply<<2)|1|(2 if self.invalid else 0))<<2


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.payload=host.validate_payload(INTEGRATION)

    def test_command_layout(self):
        rng=random.Random(20261008)
        for _ in range(128):
            a=rng.getrandbits(30)*4;v=rng.getrandbits(32);wr=rng.randrange(2)
            value,n=host.command_scan(wr,a,v)
            self.assertEqual(n,77)
            self.assertEqual(f'{value:077b}',f'{2:02b}{wr:01b}{v:032b}{a:032b}{0:08b}00')

    def test_full_transfer_readback(self):
        link=LinkModel();bus=host.Bus(link)
        rows=host.transfer(bus,self.payload)
        self.assertEqual(sum(r['bytes'] for r in rows),750016)
        self.assertEqual(len(link.writes),187504)
        self.assertNotIn(host.MB,dict(link.writes))
        self.assertEqual(bus.errors,0)
        self.assertEqual([r['crc32'] for r in rows],['c89fa72e','fc4cc6a5','7c6149bf'])

    def test_start_command_last(self):
        link=LinkModel();bus=host.Bus(link)
        result=host.start_bounded(bus,self.payload,3)
        self.assertEqual(result['status'],'BOUNDED_BOARD_RUN_PASS')
        self.assertEqual(link.writes[-1],(host.MB,0x434e4e31))
        self.assertEqual(dict(link.writes)[host.MB+12],0x7c6149bf)

    def test_bounded_memory(self):
        link=LinkModel();rows=host.memory_test(host.Bus(link))
        self.assertEqual(len(rows),2)
        self.assertTrue(all(host.SCRATCH<=a<host.SCRATCH+4096 for a,_ in link.writes))

    def test_bus_error_and_timeout(self):
        for invalid,never in ((True,False),(False,True)):
            link=LinkModel(invalid,never);bus=host.Bus(link)
            with self.assertRaises((IOError,TimeoutError)):bus.read(host.APB)
            self.assertEqual(bus.errors,1)

    def test_guard_addresses_and_expired(self):
        bus=host.Bus(LinkModel())
        for address in (host.APB+8,0xe0000000,0xe2000000,0xe1000001,host.MB+4):
            with self.assertRaises(ValueError):bus.write(address,0)
        bus.deadline=0
        with self.assertRaises(TimeoutError):bus.read(host.APB)

    def test_bad_package(self):
        raw=accepted_package()[0]
        cases=[raw[:-1],raw+b'\0',raw[:17792]+b'\x01'+raw[17793:]]
        for offset in (0,32,400,8192,17600):
            b=bytearray(raw);b[offset]^=1;cases.append(bytes(b))
        for b in cases:
            with self.assertRaises((AssertionError,ValueError,KeyError,RuntimeError)):decode(b)

    def test_stale_residual_even_with_recomputed_crc(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as td:
            root=Path(td);shutil.copy2(INTEGRATION/'result.json',root/'result.json')
            shutil.copytree(INTEGRATION/'descriptors',root/'descriptors')
            dm=read_json(root/'descriptors/result.json');schema=read_json(root/'descriptors/schema.json')
            f={r['field']:r for r in schema['compute']['fields']};program=read_json(Path(dm['program_path'])/'program.json')
            raw=bytearray((root/'descriptors/compute.bin').read_bytes())
            for k in range(0,len(raw),64):
                word=int.from_bytes(raw[k:k+64],'little');idx=(word>>f['program_index']['offset'])&((1<<f['program_index']['width'])-1)
                if program[idx]['skip']:
                    word^=1<<f['f24']['offset'];raw[k:k+64]=word.to_bytes(64,'little');break
            (root/'descriptors/compute.bin').write_bytes(raw)
            dm['files']['compute.bin']=hashlib.sha256(raw).hexdigest();dm['crc32']['compute']=f'{zlib.crc32(raw):08x}'
            write_json(root/'descriptors/result.json',dm)
            r=read_json(root/'result.json');r['descriptor_crc32']['compute']=dm['crc32']['compute'];write_json(root/'result.json',r)
            with self.assertRaisesRegex(AssertionError,'stale align multiplier'):host.validate_payload(root)

    def test_readback_corruption(self):
        link=LinkModel();bus=host.Bus(link)
        with self.assertRaisesRegex(IOError,'readback mismatch'):host.transfer(bus,self.payload,write=False)

    def test_bundle_mutation_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as td:
            root=Path(td);bit=root/'test.bit';bit.write_bytes(b'fixture-not-a-bitstream')
            for name,data in self.payload.items():(root/(name+'.bin')).write_bytes(data)
            r=dict(status='RELEASED_FOR_BOUNDED_BRINGUP',package_sha256=host.PACKAGE_SHA,
                   bitstream=str(bit),bitstream_sha256=host.sha(bit),required_evidence={},
                   bundle_files={p.name:host.sha(p) for p in root.iterdir()})
            write_json(root/'release.json',r)
            host.validate_release(root/'release.json',self.payload)
            (root/'dma.bin').write_bytes(b'bad')
            with self.assertRaises(AssertionError):host.validate_release(root/'release.json',self.payload)

    def test_installed_vendor_jtag_initialization(self):
        # Exercise the actual installed fork, while replacing only USB I/O.
        host.usb_runtime()
        from pyftdi.jtag import JtagEngine
        engine=JtagEngine(trst=False,frequency=1e6);controller=engine.controller
        controller._ftdi=Mock()
        host.configure_engine(engine,'ftdi://ftdi:4232h:TEST/2')
        controller._ftdi.open_mpsse_from_url.assert_called_once_with(
            'ftdi://ftdi:4232h:TEST/2',direction=controller.direction,frequency=1e6)
        packet=controller._ftdi.write_data.call_args.args[0]
        self.assertEqual(packet[1:],bytes([0,controller.direction]))
        controller._ftdi.set_latency_timer.assert_called_once_with(1)


if __name__=='__main__':unittest.main(verbosity=2)
