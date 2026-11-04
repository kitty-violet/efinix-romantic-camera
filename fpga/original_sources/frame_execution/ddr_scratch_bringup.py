"""Bounded scratch DDR test of the frozen local-drivers release. No model loading."""
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
import zlib
from pathlib import Path

from fixed_jtag_loader import APB, MB, SCRATCH, command_scan, decode_response
from stable_jtag_reply import StableReplyTransport
from read_startup_cause_snapshot import fresh
from windows_process_tree import stop_launched_tree

ROOT = Path('E:/fpga')
HERE = Path(__file__).resolve().parent
BUNDLE = ROOT / 'output/deployment/board-full-150-local-drivers-20261012'
BIT_SHA = 'f630349fdadfb9c3c146a5dd4682e49df97bb8f022952deb7726d0ca8daec9dc'
MANIFEST_SHA = '9c6513db755258c4a2ae5d58d47449aff3313674beb7945e63f6aec1c82bdd56'
READ_STATUS = {APB, APB+4, APB+8, APB+12, APB+0x6c, MB, MB+4}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec); spec.loader.exec_module(obj)
    return obj

class ScratchBus:
    def __init__(self, transport, seconds=110):
        self.transport = transport
        self.deadline = time.monotonic()+seconds
        self.transactions = self.errors = self.writes = self.reads = 0
        self.attempted_commands = self.attempted_writes = 0

    def access(self, writing, address, value=0):
        scratch = SCRATCH <= address < SCRATCH+4096 and address % 4 == 0
        if not scratch and (writing or address not in READ_STATUS):
            raise ValueError('Scratch-only address policy rejected request')
        value_scan, count = command_scan(writing, address, value)
        if time.monotonic() >= self.deadline: raise TimeoutError('Total DDR test deadline')
        if self.attempted_commands >= 5000: raise RuntimeError('DDR test command budget')
        self.attempted_commands += 1
        if writing: self.attempted_writes += 1
        try:
            # Exactly one command, no write retry after timeout or bus error.
            self.transport.scan(value_scan, count)
            end = min(self.deadline, time.monotonic()+2)
            for _ in range(200):
                if time.monotonic() >= end: break
                response = decode_response(self.transport.scan(1, 36))
                if response is not None:
                    self.transactions += 1
                    if writing: self.writes += 1
                    else: self.reads += 1
                    return response
            raise TimeoutError(f'No DDR/control reply at {address:08x}')
        except Exception:
            self.errors += 1
            raise

    def read(self, address): return self.access(False, address)
    def write(self, address, value): return self.access(True, address, value)

def guard(bus):
    values = {f'{a:08x}': bus.read(a) for a in sorted(READ_STATUS)}
    if values[f'{APB:08x}'] != 0x54493630: raise RuntimeError('Wrong identity')
    if values[f'{APB+4:08x}'] or values[f'{APB+8:08x}']:
        raise RuntimeError('Must remain unloaded and disabled')
    if values[f'{APB+12:08x}'] != 0xb1000000: raise RuntimeError('Pipeline error present')
    ddr = values[f'{APB+0x6c:08x}']
    if ddr & 255 != 255 or ddr >> 31 & 1:
        raise RuntimeError('DDR not qualified, reset/PLL absent, or CNN enabled')
    if values[f'{MB:08x}'] != 0 or values[f'{MB+4:08x}'] != 1:
        raise RuntimeError('Boot loader is not idle')
    return values

def patterns():
    return [('lane_probe', [0x01234567, 0x89abcdef, 0x13579bdf, 0x2468ace0]),
            ('address', [SCRATCH+4*i for i in range(1024)]),
            ('inverse_address', [(SCRATCH+4*i)^0xffffffff for i in range(1024)])]

def exercise(bus, out, save):
    out['guards'].append(guard(bus)); save()
    for name, words in patterns():
        row = dict(name=name, bytes=len(words)*4, written_words=0, checked_words=0,
                   mismatch_count=0, status='WRITING')
        out['patterns'].append(row); save(); started=time.monotonic()
        expected = b''.join(w.to_bytes(4, 'little') for w in words)
        row['expected_crc32'] = f'{zlib.crc32(expected):08x}'
        for i, word in enumerate(words):
            bus.write(SCRATCH+4*i, word); row['written_words'] += 1
            if i % 128 == 0: save()
        row['status'] = 'READING'; save(); actual=bytearray()
        for i, word in enumerate(words):
            value=bus.read(SCRATCH+4*i); actual.extend(value.to_bytes(4, 'little'))
            row['checked_words'] += 1
            if value != word:
                row.update(status='FAIL', mismatch_count=1,
                           first_mismatch=dict(address=f'{SCRATCH+4*i:08x}',
                                               expected=f'{word:08x}', actual=f'{value:08x}'))
                row['valid_read_prefix_hex'] = actual.hex(); save()
                raise IOError('DDR word mismatch; stopped at first error')
            if i % 128 == 0: save()
        row['actual_crc32'] = f'{zlib.crc32(actual):08x}'
        if row['actual_crc32'] != row['expected_crc32']: raise IOError('DDR CRC mismatch')
        row.update(status='PASS', seconds=round(time.monotonic()-started, 3)); save()
        out['guards'].append(guard(bus)); save()

def validate_evidence():
    programmer=module(BUNDLE/'program.py', 'scratch_release_validation')
    proof=programmer.validate()
    assert proof['manifest_sha256'] == MANIFEST_SHA and proof['bitstream_sha256'] == BIT_SHA
    manifest=load(BUNDLE/'manifest.json')
    startup=ROOT/'tmp/board-test/local-drivers-board-analysis1.json'
    d=load(startup)
    assert d['status']=='READ_ONLY_STARTUP_AND_SW1_RECOVERY_PASS'
    assert d['programmed_bit_sha256']==BIT_SHA and d['manifest_sha256']==MANIFEST_SHA
    for p,h in d['evidence'].items(): assert sha(p)==h
    board=Path(manifest['board'])
    paths=[board/'ddr_memory_map.vh', board/'assembly/sapphire_axi32_to_axi128.sv',
           board/'assembly/ti60_ddr_four_master_fabric.sv', board/'ddr516_axi_address_bridge.sv',
           ROOT/'tmp/frame-execution/connected-board-startup-cause4/assembly/EfxSapphireSoc.v']
    for p in paths: assert manifest['required_evidence'][str(p)]==sha(p),str(p)
    memory=paths[0].read_text()
    for name, value in [('CPU_SOURCE_BASE','e1000000'), ('CPU_SOURCE_LIMIT','e1ffffff'),
                        ('CPU_DDR_BASE','05000000'), ('CPU_DDR_LIMIT','05ffffff')]:
        assert f"`define TI60_DDR_{name} 32'h{value}" in memory
    assert 'wire ddr_hit=address[31:24]==8\'he1;' in paths[4].read_text()
    assert 'if(bus_age>=4095)fatal_bus_error<=1;' in paths[4].read_text()
    support=[HERE/n for n in ('fixed_jtag_loader.py','fixed_jtag_physical_b.py',
              'stable_jtag_reply.py','read_startup_cause_snapshot.py','read_startup_cause.py',
              'read_control_snapshot.py','windows_process_tree.py')]
    return dict(status='DDR_SCRATCH_ONLY_RELEASED', manifest_sha256=MANIFEST_SHA,
        bitstream_sha256=BIT_SHA, board_startup_sha256=sha(startup),
        source_host_range=['e1f00000','e1f00fff'], physical_ddr_range=['05f00000','05f00fff'],
        writes_limited_to_scratch=True, weights_allowed=False, program_load_allowed=False,
        cnn_start_allowed=False, flash_allowed=False, reprogram_allowed=False,
        total_host_timeout_s=120, per_reply_timeout_s=2, max_commands=5000,
        preflight_validation=proof,
        evidence={str(p):sha(p) for p in paths+support+[Path(__file__), startup]})

def worker(url, report, release):
    from fixed_jtag_physical_b import PhysicalBTransport
    out=dict(status='STARTING', hardware_access=False, weights_loaded=False,
             program_loaded=False, cnn_started=False, flash_written=False, fpga_programmed=False,
             url=url, release_sha256=sha(release), patterns=[], guards=[], scans=[], snapshots=[])
    report.parent.mkdir(parents=True,exist_ok=True)
    def save(): report.write_text(json.dumps(out,indent=2),encoding='utf-8')
    t=None; bus=None; stable=None; save()
    def observe():
        s=fresh(t,out['scans'],save);out['snapshots'].append(s);save()
        return s
    try:
        r=load(release)
        assert r['status']=='DDR_SCRATCH_ONLY_RELEASED' and r['bitstream_sha256']==BIT_SHA
        for p,h in r['evidence'].items(): assert sha(p)==h
        t=PhysicalBTransport(url);out.update(hardware_access=True,idcode=f'{t.idcode:08x}',usb_binding=t.usb_binding)
        first=observe()
        if not first['ddr']['raw_done'] or not first['ddr']['raw_pass'] or not first['levels']['ddr_qualified']:
            raise RuntimeError('Fresh independent DDR calibration guard failed')
        if first['payload']['captured'] or first['adapter']['captured']:
            raise RuntimeError('Previous first-fault capture present; no write allowed')
        if not all(p['matches'] for p in first['cell_probes'].values()):
            raise RuntimeError('Primitive probes differ from released implementation')
        stable=StableReplyTransport(t);bus=ScratchBus(stable)
        exercise(bus,out,save)
        last=observe()
        for key in ('accepted_commands','replies','received_last'):
            assert (last['live'][key]-first['live'][key])&255 == bus.transactions&255,key
        assert (last['live']['received_bits']-first['live']['received_bits'])&255 == bus.transactions*75&255
        if last['adapter']['captured'] or last['adapter']['live']['read_error'] or last['adapter']['live']['write_error']:
            raise RuntimeError('Adapter error captured during DDR test')
        # An early-stall sample alone is diagnostic; actual fatal/boot errors reject.
        flags=last['live']['flags']
        if flags['boot_fault'] or flags['signals']['fatal_bus_error'] or flags['signals']['boot_failed']:
            raise RuntimeError('Control fault after DDR test')
        if not last['ddr']['raw_done'] or not last['ddr']['raw_pass']:
            raise RuntimeError('DDR qualification lost')
        out['status']='PHYSICAL_4KIB_DDR_READ_WRITE_CRC_PASS'
    except Exception as exc:
        out.update(status='FAIL_PREFIX_SAVED',error=repr(exc))
        if t:
            try: observe()
            except Exception as later: out['failure_snapshot_error']=repr(later)
    finally:
        if bus: out.update(transactions=bus.transactions,attempted_commands=bus.attempted_commands,
            writes=bus.writes,reads=bus.reads,attempted_writes=bus.attempted_writes,transport_errors=bus.errors)
        if stable: out.update(reply_poll_pairs=stable.poll_pairs,unstable_pairs=stable.unstable_pairs)
        if t:
            try: t.close()
            except Exception as exc: out.update(status='FAIL_PREFIX_SAVED',close_error=repr(exc))
        save()
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--report',required=True,type=Path)
    p.add_argument('--release',type=Path);p.add_argument('--url')
    p.add_argument('--execute-supervised',action='store_true');p.add_argument('--worker',action='store_true')
    a=p.parse_args();assert not a.report.exists(),'Preserve existing result'
    if not a.execute_supervised:
        assert not a.worker and not a.url
        r=validate_evidence();a.report.write_text(json.dumps(r,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in r.items() if k not in ('evidence','preflight_validation')},indent=2));return
    assert a.url and a.release
    if a.worker:
        r=worker(a.url,a.report,a.release)
        raise SystemExit(r['status']!='PHYSICAL_4KIB_DDR_READ_WRITE_CRC_PASS')
    # Validate the previously prepared scope and the complete frozen package before USB access.
    assert load(a.release)==validate_evidence(),'Release changed or is stale'
    log=a.report.with_suffix('.log');assert not log.exists();started=time.monotonic()
    with log.open('x') as f:
        proc=subprocess.Popen([sys.executable,'-B',__file__,'--worker','--execute-supervised',
            '--url',a.url,'--report',str(a.report),'--release',str(a.release)],stdout=f,stderr=subprocess.STDOUT)
        try: code=proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            cleanup=stop_launched_tree(proc);code=124
            r=load(a.report) if a.report.exists() else {}
            r.update(status='HOST_TIMEOUT_PREFIX_SAVED',cleanup=cleanup)
            a.report.write_text(json.dumps(r,indent=2),encoding='utf-8')
    r=load(a.report) if a.report.exists() else dict(status='WORKER_FAILED')
    r.update(exit_code=code,host_seconds=round(time.monotonic()-started,3))
    a.report.write_text(json.dumps(r,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in r.items() if k not in ('scans','snapshots')},indent=2))
    raise SystemExit(code)

if __name__=='__main__':main()
