"""Ti60 fixed-controller host operations, without RV32 injection or Flash writes.

Only USER1 DR traffic is generated. This file never programs a bitstream.
The default action validates files offline; USB access requires an explicit URL.
"""
import argparse
import hashlib
import json
import sys
import time
import zlib
from types import SimpleNamespace
from pathlib import Path
from accepted_a_assets import ROOT, PACKAGE_SHA, accepted_package, read_json, write_json, sha

EF = Path('E:/efinity-2026.1.132-windows-x64/efinity/2026.1')
MB, APB = 0xf9000f00, 0xf8100000
WINDOWS = {'compute':(0xe1000000,562496), 'dma':(0xe1100000,167040), 'coefficients-padded':(0xe1200000,20480)}
SCRATCH = 0xe1f00000


def command_scan(write, address, value=0):
    if address & 3 or not 0 <= address < 2**32 or not 0 <= value < 2**32:
        raise ValueError('unaligned address or non-u32 data')
    # First two bits select the wrapper; dispatcher=0, size=word (2).
    command = (2 << 73) | (int(write) << 72) | (value << 40) | (address << 8)
    return command << 2, 77


def decode_response(value):
    response = value >> 2  # discard two wrapper bits from TDO
    if not response & 1:
        return None
    if response & 2:
        raise IOError('fixed controller returned a bus error')
    return (response >> 2) & 0xffffffff


def validate_payload(integration):
    raw, _, residual, _ = accepted_package()
    proof = read_json(integration/'result.json')
    if proof['package_sha256'] != PACKAGE_SHA:
        raise ValueError('package identity')
    d = integration/'descriptors'; dm = read_json(d/'result.json')
    schema = read_json(d/'schema.json')
    assert sha(d/'schema.json') == dm['schema_sha256']
    assert dm['package_sha256'] == PACKAGE_SHA
    program_path = Path(dm['program_path'])/'program.json'
    assert sha(program_path) == dm['program_sha256']
    program = read_json(program_path)
    payload = {}
    for name in ('compute','dma'):
        data = (d/(name+'.bin')).read_bytes()
        assert len(data) == WINDOWS[name][1]
        assert hashlib.sha256(data).hexdigest() == dm['files'][name+'.bin']
        assert f'{zlib.crc32(data):08x}' == dm['crc32'][name] == proof['descriptor_crc32'][name]
        payload[name] = data
    fields = {f['field']:f for f in schema['compute']['fields']}
    def get(w, field):
        f = fields[field]; return (w >> f['offset']) & ((1 << f['width'])-1)
    records = {}; data = payload['compute']
    for k in range(0,len(data),64):
        w=int.from_bytes(data[k:k+64],'little');c=program[get(w,'program_index')]
        assert c['kind']=='compute'
        if c['skip']:
            r=residual[c['node']]
            assert (get(w,'f6')>>5)&1 == int(r['main_scaled']), 'stale main_scaled'
            assert get(w,'f24') == r['multiplier'], 'stale align multiplier'
            assert get(w,'f25') == r['post_multiplier'], 'stale post multiplier'
            records[c['node']]=records.get(c['node'],0)+1
    assert records==dm['residual_records']=={'res1_add':192,'res2_add':188,'res3_add':180}
    payload['coefficients-padded']=raw
    assert f'{zlib.crc32(raw):08x}'==proof['padded_crc32']
    return payload


def usb_runtime():
    # Use installed vendor files only; keep this dependency path process-local.
    sys.path.append(str(EF/'python311/Lib/site-packages'))
    from pyftdi.jtag import JtagEngine
    from pyftdi.bits import BitSequence
    from pyftdi.usbtools import UsbTools
    try:
        UsbTools.find_backend()
    except ValueError:
        import usb.backend.libusb1 as libusb
        library=EF/'debugger/openocd/bin/libusb-1.0.dll'
        if not library.is_file() or libusb.get_backend(find_library=lambda name:str(library)) is None:
            raise RuntimeError('Installed libusb backend unavailable; no driver or software was installed')
        UsbTools.find_backend()
    return JtagEngine,BitSequence


def validate_release(path, payload):
    release=read_json(path)
    assert release['status']=='RELEASED_FOR_BOUNDED_BRINGUP' and release['package_sha256']==PACKAGE_SHA
    assert sha(release['bitstream'])==release['bitstream_sha256']
    for name,digest in release['bundle_files'].items():assert sha(path.parent/name)==digest,name
    for name,data in payload.items():assert (path.parent/(name+'.bin')).read_bytes()==data,name
    for name,digest in release['required_evidence'].items():assert sha(name)==digest,name
    return release


def jtag_init():
    # The installed Efinity fork dereferences init.val before its None check.
    # Empty fields explicitly select its normal TCK/TDI/TMS-only defaults.
    return SimpleNamespace(val='',dir='')


def configure_engine(engine,url):
    engine.configure(url,init=jtag_init())
    # This library's JTAG reads do not send SEND_IMMEDIATE. Avoid its 16ms
    # default accumulating across hundreds of thousands of short USB reads.
    # FTDI session latency only; no FPGA clock/RTL or installed files change.
    engine.controller.ftdi.set_latency_timer(1)


class PyFtdiTransport:
    def __init__(self, url):
        JtagEngine,BitSequence=usb_runtime()
        self.bits=BitSequence;self.engine=JtagEngine(trst=False,frequency=1e6)
        if not url.startswith('ftdi://') or not url.endswith('/2'):
            raise ValueError('V4 JTAG must use its explicit FT4232 B interface (/2)')
        try:
            configure_engine(self.engine,url); self.engine.reset(); self.engine.go_idle()
            self.engine.write_ir(BitSequence(3,length=5)); self.engine.go_idle()
            self.idcode=int(self.scan(0,32))
            if self.idcode not in (0x00360a79,0x10660a79):
                raise IOError(f'Unexpected Ti60 IDCODE {self.idcode:08x}; no payload written')
            self.engine.write_ir(BitSequence(8,length=5)); self.engine.go_idle(); self.engine.sync()
        except Exception:
            self.close();raise

    def scan(self, value, count):
        self.engine.change_state('shift_dr')
        result=self.engine.shift_and_update_register(self.bits(value,length=count))
        self.engine.go_idle();self.engine.sync()
        return int(result)

    def close(self):
        self.engine.close()


class Bus:
    def __init__(self, transport, seconds=1800):
        self.transport=transport;self.deadline=time.monotonic()+seconds
        self.transactions=0;self.errors=0;self.retries=0

    def access(self, writing, address, value=0):
        if time.monotonic() >= self.deadline:raise TimeoutError('host operation deadline')
        allowed = any(base <= address < base+size for base,size in WINDOWS.values())
        allowed |= SCRATCH <= address < SCRATCH+4096
        allowed |= MB <= address <= MB+0x24 if not writing else address in (MB,MB+8,MB+12,MB+16,MB+20,MB+24)
        allowed |= not writing and address in {APB+x for x in (0,4,8,0x50,0x54,0x58,0x5c,0x60,0x64,0x68)}
        if not allowed:raise ValueError(f'Host address policy rejected {address:08x}')
        scan,n=command_scan(writing,address,value)
        self.transport.scan(scan,n)
        deadline=min(self.deadline,time.monotonic()+2)
        try:
            for attempt in range(200):
                if time.monotonic()>=deadline:break
                result=decode_response(self.transport.scan(1,36))
                if result is not None:
                    self.transactions+=1;self.retries+=attempt;return result
            raise TimeoutError(f'JTAG reply timeout {address:08x}')
        except Exception:
            self.errors+=1;raise

    def read(self,address):return self.access(False,address)
    def write(self,address,value):return self.access(True,address,value)


def snapshot(bus):
    values={f'{off:02x}':bus.read(APB+off) for off in (0,4,8,0x50,0x54,0x58,0x5c,0x60,0x64)}
    assert values['00']==0x54493630,'APB hardware identity'
    return dict(phase=bus.read(MB+4),apb=values,observed_coefficient_crc=bus.read(MB+28))


def idle(bus):
    s=snapshot(bus)
    if s['phase'] != 1 or s['apb']['08'] != 0:
        raise RuntimeError('Board is not freshly configured and idle; reconfigure the approved bitstream')
    return s


def transfer(bus, payload, write=True):
    idle(bus);results=[]
    for name,data in payload.items():
        base,size=WINDOWS[name];assert size==len(data)
        started=time.monotonic()
        if write:
            for off in range(0,size,4):
                bus.write(base+off,int.from_bytes(data[off:off+4],'little'))
                if off and off%65536==0:print('UPLOAD',name,off,'/',size,flush=True)
        actual=bytearray()
        for off in range(0,size,4):
            actual.extend(bus.read(base+off).to_bytes(4,'little'))
        if actual!=data:raise IOError(f'{name} full readback mismatch')
        results.append(dict(name=name,bytes=size,crc32=f'{zlib.crc32(actual):08x}',
                            sha256=hashlib.sha256(actual).hexdigest(),seconds=time.monotonic()-started))
        print('READBACK_PASS',name,size,flush=True)
    return results


def memory_test(bus):
    idle(bus);rows=[]
    for inverse in (0,0xffffffff):
        data=b''.join(((SCRATCH+4*i)^inverse).to_bytes(4,'little') for i in range(1024))
        for i in range(1024):bus.write(SCRATCH+4*i,int.from_bytes(data[4*i:4*i+4],'little'))
        actual=b''.join(bus.read(SCRATCH+4*i).to_bytes(4,'little') for i in range(1024))
        if actual!=data:raise IOError('bounded DDR memory test mismatch')
        rows.append(dict(bytes=4096,crc32=f'{zlib.crc32(actual):08x}'))
    return rows


def start_bounded(bus,payload,frames=1,tag=0xa008):
    if not 1<=frames<=16 or not 1<=tag<=65535:raise ValueError('frame budget or tag')
    before=idle(bus);base=before['apb']['5c'];start=time.monotonic()
    for off,value in ((8,tag),(12,zlib.crc32(payload['coefficients-padded'])),
                      (16,zlib.crc32(payload['compute'])),(20,zlib.crc32(payload['dma'])),(24,frames)):
        bus.write(MB+off,value)
    # The only run command is sent after all payload readback and parameters.
    bus.write(MB,0x434e4e31)
    phases=[];samples=[]
    while time.monotonic()-start<15:
        s=snapshot(bus);samples.append(dict(host_seconds=time.monotonic()-start,**s))
        if not phases or s['phase']!=phases[-1]:
            phases.append(s['phase']);print('BOARD_PHASE',hex(s['phase']),flush=True)
        if s['phase']&0x80000000 or s['apb']['04']&24:
            raise IOError(f'board failed: {s}')
        if s['phase']==5:
            assert s['apb']['08']==0 and s['observed_coefficient_crc']==zlib.crc32(payload['coefficients-padded'])
            # STOP clears both enable and compare_mode.
            assert (s['apb']['5c']-base)&0xffffffff >= frames
            return dict(status='BOUNDED_BOARD_RUN_PASS',phases=phases,samples=samples,
                        last_cnn_cycles=s['apb']['58'],committed=s['apb']['5c']-base,
                        not_end_to_end_fps=True)
        time.sleep(.05)
    raise TimeoutError('No terminal phase in 15s; hardware has independent 5s run timeout. Reconfigure before retry.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--integration',type=Path,default=ROOT/'tmp/frame-execution/accepted-a-integration1')
    p.add_argument('--action',choices=['validate','devices','inspect','memtest','load','run'],default='validate')
    p.add_argument('--url');p.add_argument('--frames',type=int,default=1);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--release',type=Path);p.add_argument('--prior-report',type=Path);a=p.parse_args()
    assert not a.report.exists(),'Preserve existing board record'
    a.report.parent.mkdir(parents=True,exist_ok=True)
    result=dict(action=a.action,actual_board=False,status='STARTING',url=a.url);transport=None;bus=None
    try:
        payload=validate_payload(a.integration.resolve())
        result['files']={k:dict(bytes=len(v),sha256=hashlib.sha256(v).hexdigest(),crc32=f'{zlib.crc32(v):08x}') for k,v in payload.items()}
        if a.release:
            release=validate_release(a.release.resolve(),payload)
            result['release_sha256']=sha(a.release)
        if a.action=='validate':result['status']='ACCEPTED_A_DEPLOYMENT_PAYLOAD_VALID'
        elif a.action=='devices':
            usb_runtime()
            from pyftdi.ftdi import Ftdi
            result['devices']=[dict(device=d._asdict(),interfaces=n) for d,n in Ftdi.list_devices('ftdi://ftdi:4232h/?')]
            result['status']='USB_ENUMERATION_COMPLETE'
        else:
            if not a.url or not a.release:raise ValueError('Explicit physical URL and release receipt are required')
            prior=None
            if a.action in ('load','run'):
                if not a.prior_report:raise ValueError('Prior memory-test/load receipt is required')
                prior=read_json(a.prior_report)
                assert prior['status']==('PHYSICAL_MEMTEST_PASS' if a.action=='load' else 'PHYSICAL_LOAD_PASS')
                assert prior['url']==a.url and prior['files']==result['files']
                result['prior_report_sha256']=sha(a.prior_report)
            time.sleep(1)  # stable clocks/reset guards after a freshly configured device
            transport=PyFtdiTransport(a.url);bus=Bus(transport);result['actual_board']=True
            result['idcode']=f'{transport.idcode:08x}'
            if prior:assert prior['idcode']==result['idcode']
            if a.action=='inspect':result['snapshot']=snapshot(bus)
            elif a.action=='memtest':result['memory_test']=memory_test(bus)
            elif a.action=='load':result['readback']=transfer(bus,payload)
            else:
                result['readback']=transfer(bus,payload,write=False)
                result['run']=start_bounded(bus,payload,a.frames)
            result['status']='PHYSICAL_'+a.action.upper()+'_PASS'
    except Exception as e:
        result.update(status='FAIL',error=repr(e));raise
    finally:
        if bus:result.update(transactions=bus.transactions,transport_errors=bus.errors,response_poll_retries=bus.retries)
        if transport:transport.close()
        write_json(a.report,result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
