"""Prepare INPUT_PREVIEW using resident coefficient CRC checks; never start CNN.

This does not certify retained DDR compute/DMA images after reconfiguration.
LOAD never runs compute; select/read back the preview bypass after LOAD reaches
terminal WAIT_CMD, the actual mode-change prerequisite. No DDR/Flash writes.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
from run_units import ROOT, sha
from video_service_board import bound, ControlBus, diagnostic_once
from video_service_control import prepare_once
from video_service_observation import fresh, prepared, path_status
from fixed_jtag_loader import validate_payload, snapshot
from windows_process_tree_safe import stop_launched_tree


def worker(a):
    m, receipt = bound(a.manifest, a.program_receipt)
    data = validate_payload(Path(m['a_integration']))
    if any((a.manifest.parent/(n+'.bin')).read_bytes()!=b for n,b in data.items()):
        raise ValueError('Accepted payload differs')
    out = dict(status='PREPARING_INPUT_PREVIEW', manifest_sha256=sha(a.manifest),
               program_receipt_sha256=sha(a.program_receipt), helper_sha256=sha(__file__),
               scans=[], diagnostic={}, load={}, cnn_started=False, flash_written=False,
               ddr_data_writes=0, compute_dma_ddr_reverified=False)
    def save(): a.report.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    save(); link=bus=None
    try:
        from fixed_jtag_physical_b import PhysicalBTransport
        from stable_jtag_reply import StableReplyTransport
        from ddr_scratch_bringup import guard
        link=PhysicalBTransport(receipt['url'])
        bus=ControlBus(StableReplyTransport(link),writing=True,diagnostic=True,seconds=110)
        out['guard']=guard(bus); save()
        def path():
            for _ in range(112 if m['diagnostic_schema']==6 else 96):
                v=path_status(fresh(link,m['service_diagnostic_positions'],m['camera_diagnostic_slots'],out['scans'],save))
                if v is not None:return v
            raise TimeoutError('No current path status')
        ready=lambda:prepared(link,m['service_diagnostic_positions'],m['camera_diagnostic_slots'],out['scans'],save)
        prepare_once(bus,data,out['load'],save,read_prepared=ready)
        selected=diagnostic_once(bus,out['diagnostic'],save,read_path=path,mode=1)
        if selected['mode']!=1:raise RuntimeError('INPUT_PREVIEW bypass not selected')
        out['after']=snapshot(bus); out['path']=path()
        if out['path']['mode']!=1 or out['after']['phase']!=5 or out['after']['apb']['08']:
            raise RuntimeError('Preview mode/preparation changed')
        out['status']='INPUT_PREVIEW_PREPARED_NOT_RUNNING_NOT_CNN_RELEASE'
    except Exception as exc:
        out.update(status='FAIL_PREFIX_SAVED',error=repr(exc));raise
    finally:
        if bus:out['control']=dict(transactions=bus.transactions,errors=bus.errors,attempts=bus.attempts)
        if link:link.close()
        save()


def main():
    p=argparse.ArgumentParser()
    for n in ('manifest','program-receipt','report'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--supervised',action='store_true')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);a=p.parse_args()
    a.manifest=a.manifest.resolve();a.program_receipt=a.program_receipt.resolve();a.report=a.report.resolve()
    bound(a.manifest,a.program_receipt)
    if not a.execute:print('PREVIEW_PREPARE_OFFLINE_BOUND_NO_USB');return
    if not a.supervised or not a.report.is_relative_to(ROOT/'tmp/board-test') or a.report.exists():
        raise ValueError('New supervised board-test report required')
    if a.worker:worker(a);return
    cmd=[sys.executable,'-u',__file__,'--manifest',str(a.manifest),'--program-receipt',str(a.program_receipt),
         '--report',str(a.report),'--execute','--supervised','--worker']
    with a.report.with_suffix('.log').open('x') as log:
        child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);start=time.monotonic()
        try:rc=child.wait(timeout=150);result=dict(exit_code=rc,seconds=time.monotonic()-start)
        except subprocess.TimeoutExpired:
            rc=124;result=dict(exit_code=rc,cleanup=stop_launched_tree(child))
    a.report.with_suffix('.host.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result));raise SystemExit(rc)


if __name__=='__main__':main()
