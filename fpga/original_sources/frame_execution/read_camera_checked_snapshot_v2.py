"""Separate full offline evidence validation from bounded read-only hardware IO."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from read_camera_checked_snapshot import checked_manifest,fresh,SnapshotPending,sha
from windows_process_tree import stop_launched_tree

def save_report(path,data):
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(data,indent=2),encoding='utf8')
    for attempt in range(10):
        try:temporary.replace(path);break
        except PermissionError:
            if attempt==9:raise
            time.sleep(.01)

def worker(a):
    r=dict(status='VERIFYING_OFFLINE_EVIDENCE',hardware_access=False,read_only=True,
           ddr_accesses=0,fpga_programmed=False,scans=[],snapshots=[])
    def save():save_report(a.report,r)
    save();transport=None;started=time.monotonic()
    try:
        # All original hashes/fields are checked once, before opening USB.
        m=checked_manifest(a.manifest.resolve())
        r.update(validation_seconds=round(time.monotonic()-started,3),
                 manifest_sha256=sha(a.manifest),expected_bit_sha256=m['bitstream_sha256'],
                 status='READING',hardware_access=True)
        save()
        from fixed_jtag_physical_b import PhysicalBTransport
        transport=PhysicalBTransport(a.url)
        r.update(idcode=f'{transport.idcode:08x}',usb_binding=transport.usb_binding)
        for _ in range(3):
            r['snapshots'].append(fresh(transport,m['camera_diagnostic_slots'],r['scans'],save));save();time.sleep(.1)
        r['status']='CAMERA_SNAPSHOT_OBSERVED_NOT_FULL_BOARD_PASS'
    except SnapshotPending as e:r.update(status='SNAPSHOT_PENDING_NOT_BOARD_PASS',error=str(e))
    except Exception as e:r.update(status='FAIL',error=repr(e))
    finally:
        if transport:
            try:transport.close()
            except Exception as e:r.update(status='FAIL',close_error=repr(e))
        save()
    return int(r['status']=='FAIL')

def supervise(command,report,log,validation_budget=120,io_budget=25):
    assert not report.exists() and not log.exists()
    started=time.monotonic();io_started=None;timed_out=None;cleanup=None
    with log.open('x') as f:
        proc=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
        while proc.poll() is None:
            now=time.monotonic()
            current=json.loads(report.read_text()) if report.exists() else {}
            if io_started is None and current.get('hardware_access'):
                io_started=now
            if now-started>=validation_budget+io_budget:timed_out='total'
            elif io_started is None and now-started>=validation_budget:timed_out='offline_validation'
            elif io_started is not None and now-io_started>=io_budget:timed_out='hardware_io'
            if timed_out:
                cleanup=stop_launched_tree(proc);break
            try:proc.wait(timeout=.05)
            except subprocess.TimeoutExpired:pass
    r=json.loads(report.read_text()) if report.exists() else dict(status='WORKER_FAILED',hardware_access=False)
    code=124 if timed_out else proc.returncode
    if timed_out:r.update(status='HOST_TIMEOUT',timeout_phase=timed_out,cleanup=cleanup)
    r.update(exit_code=code,host_seconds=round(time.monotonic()-started,3),
             validation_budget_seconds=validation_budget,hardware_io_budget_seconds=io_budget)
    save_report(report,r)
    return code,r

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--url');p.add_argument('--report',type=Path)
    p.add_argument('--execute',action='store_true');p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    if not a.execute:
        checked_manifest(a.manifest.resolve());print('OFFLINE_CAMERA_READER_BINDING_VERIFIED_NO_HARDWARE_ACCESS');return
    assert a.url and a.report and not a.report.exists()
    a.report=a.report.resolve();a.report.parent.mkdir(parents=True,exist_ok=True)
    if a.worker:raise SystemExit(worker(a))
    command=[sys.executable,str(Path(__file__).resolve()),'--worker','--execute',
             '--manifest',str(a.manifest.resolve()),'--url',a.url,'--report',str(a.report)]
    code,r=supervise(command,a.report,a.report.with_suffix('.log'))
    print(json.dumps(r,indent=2));raise SystemExit(code)

if __name__=='__main__':main()
