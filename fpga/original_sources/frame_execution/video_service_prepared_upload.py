"""Upload exact A images after a verified preview STOP, without reconfiguration.

Reuse the released transfer bytecode in a private namespace with an explicit
prepared-state guard. Do not mutate the frozen module or relax its cold guard.
No LIVE, mode/ISP writes, scratch writes, programming or Flash access.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from types import FunctionType, SimpleNamespace

import accepted_a_ddr_stage as stage
from fixed_jtag_loader import validate_payload
from run_units import ROOT, sha
from video_service_board import bound, ControlBus
from video_service_control import load_plan, prepare_once
from video_service_input_isp_prepared import prepared_guard
from video_service_observation import fresh, prepared
from windows_process_tree_safe import stop_launched_tree


def transfer_checked(bus, data, out, save, *, guard):
    load_plan(data)  # Exact lengths, three program CRCs and accepted A SHA.
    namespace = dict(stage.transfer.__globals__)
    namespace['scratch'] = SimpleNamespace(guard=lambda unused_bus: guard())
    transfer = FunctionType(stage.transfer.__code__, namespace, 'prepared_transfer')
    return transfer(bus, data, out, save)


def require_idle_words(words):
    status, owners, path, epoch = (words[k] for k in ('status', 'owners', 'path', 'epoch'))
    if not status & 32 or status & 0xcf:
        raise ValueError('Frame work not drained or owner error')
    if any((owners >> (3*i)) & 7 in (1, 2, 3) for i in range(3)):
        raise ValueError('Capture or compute still owns a frame')
    if path >> 6 or path & 4 or (path & 3) not in (0, 1):
        raise ValueError('Inspection alias must be disabled and mode known')
    if epoch == 65535:
        raise ValueError('Saturated reset epoch cannot bind upload')


def tests_bound(path):
    r = json.loads(path.read_text(encoding='utf-8'))
    files = [Path(__file__), Path(__file__).with_name('test_video_service_prepared_upload.py'),
             Path(stage.__file__), Path(__file__).with_name('video_service_input_isp_prepared.py')]
    if r.get('status') != 'PASS' or r.get('tests', 0) < 8 or r.get('sources') != {str(p.resolve()): sha(p) for p in files}:
        raise ValueError('Current prepared-upload test binding required')


def worker(a):
    m, receipt = bound(a.manifest, a.program_receipt)
    if m['diagnostic_schema'] != 6:
        raise ValueError('Schema6 required')
    data = validate_payload(Path(m['a_integration']))
    load_plan(data)
    if any((a.manifest.parent/(n+'.bin')).read_bytes() != b for n,b in data.items()):
        raise ValueError('Payload differs from released bundle')
    out = dict(status='CONNECTING', manifest_sha256=sha(a.manifest),
        program_receipt_sha256=sha(a.program_receipt), helper_sha256=sha(__file__),
        test_receipt_sha256=sha(a.test_receipt), images=[], guards=[], chunks=[],
        load={}, hardware_access=False, cnn_started=False, flash_written=False,
        isp_changed=False, reprogrammed=False)
    def save():
        a.report.write_text(json.dumps(out, indent=2)+'\n', encoding='utf-8')
    save()
    link = bus = control = stable = None
    try:
        from fixed_jtag_physical_b import PhysicalBTransport
        from stable_jtag_reply import StableReplyTransport
        link = PhysicalBTransport(receipt['url']); out['hardware_access'] = True
        stable = StableReplyTransport(link)
        control = ControlBus(stable, writing=True, seconds=1830)
        bus = stage.StageBus(stable, seconds=1770)
        def guard():
            return prepared_guard(control)
        def idle():
            words = {}
            wanted = {('process', 0):'status', ('process', 1):'owners',
                      ('path', 14):'path', ('health', 13):'epoch'}
            for _ in range(224):
                row = fresh(link, m['service_diagnostic_positions'], m['camera_diagnostic_slots'], [], lambda:None)
                if not row.get('measurement_valid') or row.get('diagnostic_profile') != 'video-service-pages-v6':
                    raise ValueError('Invalid diagnostic chunk')
                key = wanted.get((row['page_kind'], row['chunk_index']))
                if key:
                    value = row['chunk_words'][0]
                    words[key] = value >> 16 if key == 'epoch' else value
                    out['chunks'].append(row)
                if len(words) == 4:
                    require_idle_words(words); return words
            raise TimeoutError('Missing drained/inspection/epoch evidence')
        out['guard_before'] = guard()
        out['idle_before'] = idle(); guard(); save()
        out['status'] = 'STAGING_THREE_IMAGES_STOPPED'; save()
        transfer_checked(bus, data, out, save, guard=guard)
        out['idle_after'] = idle(); guard()
        if out['idle_after'] != out['idle_before']:
            raise ValueError('Stopped state or reset epoch changed during transfer')
        if len(out['images']) != 3 or any(r['status'] != 'PASS' for r in out['images']):
            raise ValueError('Three complete image readbacks required')
        out['status'] = 'THREE_IMAGES_VERIFIED_PREPARING'; save()
        ready = lambda: prepared(link, m['service_diagnostic_positions'], m['camera_diagnostic_slots'], [], lambda:None)
        prepare_once(control, data, out['load'], save, read_prepared=ready)
        out['guard_after'] = guard()
        out['status'] = 'ACCEPTED_A_ALL_IMAGES_AND_RESIDENT_READY_STOPPED'; save()
    except Exception as e:
        out.update(status='FAIL_PREFIX_SAVED_NO_START', error=repr(e)); raise
    finally:
        if bus:
            out['stage_counts'] = {k:getattr(bus,k) for k in ('transactions','errors','writes','reads','attempted_commands','attempted_writes')}
        if control:
            out['control'] = dict(transactions=control.transactions, errors=control.errors, attempts=control.attempts)
        if stable:
            out['reply_poll_pairs'] = stable.poll_pairs
            out['unstable_pairs'] = stable.unstable_pairs
        if link: link.close()
        save()


def main():
    p = argparse.ArgumentParser()
    for n in ('manifest','program-receipt','report','test-receipt'):
        p.add_argument('--'+n, type=Path, required=True)
    p.add_argument('--execute', action='store_true'); p.add_argument('--supervised', action='store_true')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args()
    for k in ('manifest','program_receipt','report','test_receipt'):
        setattr(a,k,getattr(a,k).resolve())
    bound(a.manifest,a.program_receipt); tests_bound(a.test_receipt)
    if not a.execute:
        print('PREPARED_UPLOAD_OFFLINE_BOUND_NO_USB'); return
    if not a.supervised or not a.report.is_relative_to(ROOT/'tmp/board-test') or a.report.exists():
        raise ValueError('New supervised project report required')
    if a.worker: worker(a); return
    cmd = [sys.executable, '-u', __file__]
    for n in ('manifest','program-receipt','report','test-receipt'):
        cmd += ['--'+n,str(getattr(a,n.replace('-','_')))]
    cmd += ['--execute','--supervised','--worker']
    start = time.monotonic()
    with a.report.with_suffix('.log').open('x') as log:
        child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        try:
            rc = child.wait(timeout=1900); result=dict(exit_code=rc,seconds=time.monotonic()-start)
        except subprocess.TimeoutExpired:
            rc=124; result=dict(exit_code=rc,cleanup=stop_launched_tree(child))
    a.report.with_suffix('.host.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result)); raise SystemExit(rc)


if __name__ == '__main__': main()
