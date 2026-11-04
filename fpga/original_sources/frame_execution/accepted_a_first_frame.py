"""One retained A image, one run command, existing hardware auto-stop. No upload."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import accepted_a_ddr_stage as stage
import ddr_scratch_bringup as scratch
import fixed_jtag_loader as host
from stable_jtag_reply import StableReplyTransport
from windows_process_tree import stop_launched_tree

ROOT, HERE = stage.ROOT, stage.HERE
PRIOR = ROOT/'tmp/board-test/local-drivers-a-stage1.json'
ANALYSIS = ROOT/'tmp/board-test/local-drivers-a-stage-analysis1.json'
STAGE_RELEASE = ROOT/'tmp/board-test/local-drivers-a-stage-release2.json'
TAG = 0xa008
sha, load = stage.sha, stage.load


def plan(payload):
    return [(host.MB+8,TAG), (host.MB+12,int(stage.metadata(payload['coefficients-padded'])['crc32'],16)),
            (host.MB+16,int(stage.metadata(payload['compute'])['crc32'],16)),
            (host.MB+20,int(stage.metadata(payload['dma'])['crc32'],16)),
            (host.MB+24,1), (host.MB,0x434e4e31)]


class RunBus(host.Bus):
    def __init__(self, transport, payload):
        super().__init__(transport, seconds=40)
        self.plan = plan(payload)
        self.armed = False
        self.write_attempts = []
        self.attempts = 0

    def access(self, writing, address, value=0):
        if self.attempts >= 5000:
            raise RuntimeError('One-frame command budget exceeded')
        if writing:
            index = len(self.write_attempts)
            if not self.armed or index >= len(self.plan) or (address,value) != self.plan[index]:
                raise ValueError('Only the fixed five parameters and ONE final run command are allowed')
            # Consume once even if the reply is lost. Never repeat a run/write.
            self.write_attempts.append((address,value))
        self.attempts += 1
        return super().access(writing,address,value)


def continuity(current, previous):
    stage.check_snapshot(current)
    if current['live']['boot_state'] != 7:
        raise RuntimeError('Loader is not waiting for its first command')
    for key in ('sampled_system_reset_events','sampled_ddr_lock_drops','raw_status'):
        if current['ddr'][key] != previous['ddr'][key]:
            raise RuntimeError('DDR reset/calibration changed since full readback')
    for key in ('accepted_commands','replies','received_last','received_bits'):
        if current['live'][key] != previous['live'][key]:
            raise RuntimeError('Host command history changed since full readback')


def retention(bus, payload):
    rows = []
    for name in stage.ORDER:
        data = payload[name]; base = host.WINDOWS[name][0]
        offsets = sorted(set(range(0,len(data),4096)) | {len(data)-4})
        for off in offsets:
            if bus.read(base+off) != int.from_bytes(data[off:off+4],'little'):
                raise IOError(f'Retained image differs: {name}+{off:x}')
        rows.append(dict(name=name,checked_words=len(offsets),
                         scope='page sentinels plus prior complete readback; not a new full readback'))
    return rows


def run_once(bus, payload, out, save):
    before = host.idle(bus)
    out['before'] = before; out['samples'] = []; save()
    bus.armed = True
    for address,value in bus.plan[:-1]:
        bus.write(address,value)
        if bus.read(address) != value:
            raise IOError('Run parameter readback differs; START not sent')
    bus.write(*bus.plan[-1])
    out['start_command_acknowledged'] = True; save()
    started = time.monotonic()
    while time.monotonic()-started < 15:
        s = host.snapshot(bus)
        s['elapsed_host_s'] = round(time.monotonic()-started,6)
        out['samples'].append(s); save()
        if s['phase'] & 0x80000000 or s['apb']['04'] & 24:
            raise RuntimeError(f"Boot/store/pipeline failure: phase={s['phase']:08x} status={s['apb']['04']:08x}")
        if s['phase'] == 5:
            count = (s['apb']['5c']-before['apb']['5c']) & 0xffffffff
            if count != 1 or s['apb']['08'] != 0:
                raise RuntimeError('One-frame budget or automatic STOP did not hold')
            if s['observed_coefficient_crc'] != bus.plan[1][1]:
                raise RuntimeError('Hardware coefficient CRC differs')
            if (s['apb']['04']>>8)&4095 != 2560 or not (s['apb']['04']&1):
                raise RuntimeError('Resident image is incomplete or not verified')
            if s['apb']['64'] != TAG or s['apb']['50'] != bus.plan[2][1] or s['apb']['54'] != bus.plan[3][1]:
                raise RuntimeError('Hardware tag or descriptor CRC configuration differs')
            if not s['apb']['58']:
                raise RuntimeError('Missing CNN cycle measurement')
            out.update(resident_verified_by_firmware=True, committed_frames=count,
                       cnn_cycles=s['apb']['58'], cnn_ms_at_150mhz=s['apb']['58']/150000,
                       not_end_to_end_fps=True)
            return
        time.sleep(.02)
    raise TimeoutError('15s host run observation expired; no second run command')


def validate():
    analysis = load(ANALYSIS)
    assert analysis['status'] == 'PHYSICAL_A_DDR_STAGING_EVIDENCE_PASS'
    for path,digest in analysis['evidence'].items(): assert sha(path)==digest,path
    assert load(STAGE_RELEASE)==stage.validate_evidence()
    payload = stage.payload_checked()
    manifest = load(scratch.BUNDLE/'manifest.json')
    sources = [ROOT/'tmp/frame-execution/connected-board-startup-cause4/assembly/EfxSapphireSoc.v',
               ROOT/'tmp/frame-execution/connected-board-local-drivers1/assembly/cnn_apb_control.sv',
               ROOT/'tmp/frame-execution/connected-board-local-drivers1/assembly/frame_program_fetch.sv',
               ROOT/'tmp/frame-execution/accepted-a-integration1/jtag-loader2/result.json',
               ROOT/'tmp/frame-execution/control-state-lockstep-a1/result.json']
    for path in sources: assert manifest['required_evidence'][str(path)] == sha(path),path
    boot = sources[0].read_text()
    for marker in ('RUN_TIMEOUT_CYCLES=750000000','BUS_TIMEOUT_CYCLES=4096',
                   "index==2559 ? 3:1", "GET_FRAMES:if(rsp_data<1 || rsp_data>16)reject(1)"):
        assert marker in boot,marker
    files = sources+[Path(__file__),HERE/'test_accepted_a_first_frame.py',PRIOR,ANALYSIS,STAGE_RELEASE]
    return dict(status='ACCEPTED_A_RETAINED_ONE_FRAME_RELEASED',
        bitstream_sha256=scratch.BIT_SHA, manifest_sha256=scratch.MANIFEST_SHA,
        prior_full_readback_sha256=sha(PRIOR), payload=load(STAGE_RELEASE)['payload'],
        writes=[dict(address=f'{a:08x}',value=f'{v:08x}') for a,v in plan(payload)],
        frame_budget=1, hardware_run_timeout_s=5, host_operation_timeout_s=120,
        reprogram_allowed=False, ddr_host_writes_allowed=False, flash_allowed=False,
        evidence={str(p):sha(p) for p in files})


def worker(report, release):
    from fixed_jtag_physical_b import PhysicalBTransport
    out = dict(status='PREFLIGHT', hardware_access=False, fpga_programmed=False,
               ddr_host_writes=0, flash_written=False, snapshots=[], scans=[], stages=[],
               start_command_acknowledged=False, release_sha256=sha(release))
    bus = link = stable = None
    def save(): report.write_text(json.dumps(out,indent=2),encoding='utf-8')
    def observe():
        s=scratch.fresh(link,out['scans'],save);out['snapshots'].append(s);save();return s
    save()
    try:
        r=load(release)
        for path,digest in r['evidence'].items(): assert sha(path)==digest,path
        payload=stage.payload_checked();prior=load(PRIOR)
        receipts=[]
        for label,mode in [('control-before','control'),('id','id'),('control-after','control')]:
            receipt=report.with_name(report.stem+'-'+label+'.json');log=receipt.with_suffix('.log')
            assert not receipt.exists() and not log.exists()
            out['hardware_access']=True;save()
            with log.open('xb') as f:
                child=subprocess.Popen([sys.executable,'-B',str(ROOT/'tmp/board-test/probe-official-ftdi-session.py'),str(receipt),mode],stdout=f,stderr=subprocess.STDOUT)
                try: code=child.wait(timeout=22)
                except subprocess.TimeoutExpired:
                    out['cleanup']=stop_launched_tree(child);save();raise
            if code: raise RuntimeError('Fresh USB probe failed: '+label)
            receipts.append(load(receipt));out['stages'].append(dict(stage=label,status='PASS'));save()
        url=scratch.module(ROOT/'tmp/board-test/fresh-session-guard.py','run_fresh').validated_url(*receipts)
        if url!=prior['url']: raise RuntimeError('USB session changed; retained upload requires review')
        link=PhysicalBTransport(url);out.update(url=url,idcode=f'{link.idcode:08x}',usb_binding=link.usb_binding)
        stable=StableReplyTransport(link)
        current=observe();continuity(current,prior['snapshots'][-1])
        status=scratch.ScratchBus(stable);out['idle_guard']=scratch.guard(status);save()
        bus=RunBus(stable,payload);out['retention']=retention(bus,payload);save()
        out['status']='LOADING_RESIDENT_AND_RUNNING_ONE_FRAME';save()
        run_once(bus,payload,out,save)
        last=observe();stage.check_snapshot(last)
        if last['live']['flags']['signals']['boot_failed'] or not last['live']['flags']['signals']['boot_terminal']:
            raise RuntimeError('Firmware did not reach successful terminal stop')
        out['status']='PHYSICAL_A_RESIDENT_AND_ONE_FRAME_PASS'
    except Exception as exc:
        out.update(status='FAIL_PREFIX_SAVED',error=repr(exc))
        if link:
            try:
                observe()
                # Best-effort diagnostic reads only; never resend the command.
                detail=scratch.ScratchBus(stable or link,seconds=5)
                out['failure_status']={f'{a:08x}':detail.read(a) for a in sorted(scratch.READ_STATUS)}
                if bus:
                    out['failure_mailbox']={f'{off:02x}':bus.read(host.MB+off) for off in (4,28,32,36)}
            except Exception as later: out['failure_diagnostics_error']=repr(later)
    finally:
        if bus: out.update(transactions=bus.transactions,transport_errors=bus.errors,
                           write_attempts=bus.write_attempts,command_attempts=bus.attempts)
        if stable: out.update(reply_poll_pairs=stable.poll_pairs,unstable_pairs=stable.unstable_pairs)
        if link:
            try: link.close()
            except Exception as exc: out.update(status='FAIL_PREFIX_SAVED',close_error=repr(exc))
        save()
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',required=True,type=Path)
    p.add_argument('--release',type=Path);p.add_argument('--execute-supervised',action='store_true')
    p.add_argument('--worker',action='store_true');a=p.parse_args()
    assert not a.report.exists(),'Preserve previous result'
    if not a.execute_supervised:
        assert not a.worker
        r=validate();a.report.write_text(json.dumps(r,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in r.items() if k!='evidence'},indent=2));return
    assert a.release
    if a.worker:
        r=worker(a.report,a.release)
        raise SystemExit(r['status']!='PHYSICAL_A_RESIDENT_AND_ONE_FRAME_PASS')
    assert load(a.release)==validate()
    started=time.monotonic();log=a.report.with_suffix('.log');assert not log.exists()
    with log.open('xb') as f:
        child=subprocess.Popen([sys.executable,'-B',__file__,'--worker','--execute-supervised',
            '--release',str(a.release),'--report',str(a.report)],stdout=f,stderr=subprocess.STDOUT)
        try: code=child.wait(timeout=120)
        except subprocess.TimeoutExpired:
            cleanup=stop_launched_tree(child);code=124
            r=load(a.report) if a.report.exists() else {}
            r.update(status='HOST_TIMEOUT_PREFIX_SAVED',cleanup=cleanup)
            a.report.write_text(json.dumps(r,indent=2),encoding='utf-8')
    r=load(a.report) if a.report.exists() else dict(status='WORKER_FAILED')
    r.update(exit_code=code,host_seconds=round(time.monotonic()-started,3))
    a.report.write_text(json.dumps(r,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in r.items() if k not in ('scans','snapshots','samples')},indent=2))
    raise SystemExit(code)


if __name__=='__main__':main()
