"""Bounded OFP2 + stable APB reads; preserve progress even on a bus timeout.

Read only. No DDR data request, start command, programming or Flash operation.
This script is for a separately released OFP2 bitstream under supervision.
"""
import argparse,json,subprocess,sys,time
from pathlib import Path
from read_control_progress import fresh
from read_bringup_status import ReadOnlyBus,OFFSETS,decode
from stable_jtag_reply import StableReplyTransport
from fixed_jtag_loader import APB
from windows_process_tree import stop_launched_tree

def inspect(transport,out,save):
    stable=StableReplyTransport(transport);bus=ReadOnlyBus(stable)
    def snap():
        out['snapshots'].append(fresh(transport,out['scans'],save));save()
    snap()
    if out['snapshots'][-1]['levels']['control_reset']:
        raise RuntimeError('Control held in reset; no functional request sent')
    try:
        for _ in range(3):
            sample={'registers':{}};out['samples'].append(sample)
            for offset in OFFSETS:
                value=bus.read(APB+offset);sample['registers'][f'{offset:02x}']=value;save()
                if offset==0 and value!=0x54493630:raise ValueError('Identity mismatch')
            sample['decoded']=decode(sample['registers']['0c'],sample['registers']['6c']);save()
        out['status']='OFP2_AND_STABLE_STATUS_READ_COMPLETE_NOT_BOARD_PASS'
    finally:
        out.update(transactions=bus.transactions,reply_poll_pairs=stable.poll_pairs,unstable_pairs=stable.unstable_pairs)
        save()
        # A functional-bus timeout does not block the independent snapshot.
        try:
            snap()
            if out['status']=='OFP2_AND_STABLE_STATUS_READ_COMPLETE_NOT_BOARD_PASS':
                before,after=(out['snapshots'][i]['live'] for i in (0,-1))
                expected={'accepted_commands':bus.transactions,'replies':bus.transactions,
                          'received_last':bus.transactions,'received_bits':bus.transactions*75}
                delta={name:(after[name]-before[name])&255 for name in expected}
                out['progress_delta']=delta
                if delta!={name:value&255 for name,value in expected.items()}:
                    raise RuntimeError('Actual command/reply progress does not match read transactions')
                save()
        except Exception as exc:
            out.update(status='FAIL_PREFIX_SAVED',final_snapshot_error=repr(exc));save()

def worker(url,report):
    from fixed_jtag_physical_b import PhysicalBTransport
    out=dict(status='READING',hardware_access=True,read_only=True,ddr_payload_accesses=0,
             writes=0,flash_written=False,fpga_programmed=False,url=url,scans=[],snapshots=[],samples=[])
    def save():report.write_text(json.dumps(out,indent=2),encoding='utf-8')
    t=None;save()
    try:
        t=PhysicalBTransport(url);out.update(idcode=f'{t.idcode:08x}',usb_binding=t.usb_binding)
        inspect(t,out,save)
    except Exception as exc:out.update(status='FAIL_PREFIX_SAVED',error=repr(exc))
    finally:
        if t:
            try:t.close()
            except Exception as exc:out.update(status='FAIL_PREFIX_SAVED',close_error=repr(exc))
        save()
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);a=p.parse_args()
    assert not a.report.exists();a.report.parent.mkdir(parents=True,exist_ok=True)
    if a.worker:raise SystemExit(worker(a.url,a.report)['status']!='OFP2_AND_STABLE_STATUS_READ_COMPLETE_NOT_BOARD_PASS')
    log=a.report.with_suffix('.log');assert not log.exists();started=time.monotonic()
    with log.open('x') as f:
        child=subprocess.Popen([sys._base_executable,__file__,'--worker','--url',a.url,'--report',str(a.report.resolve())],stdout=f,stderr=subprocess.STDOUT)
        try:code=child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            cleanup=stop_launched_tree(child);code=124
            prefix=json.loads(a.report.read_text()) if a.report.exists() else {}
            prefix.update(status='HOST_TIMEOUT_PREFIX_SAVED',cleanup=cleanup);a.report.write_text(json.dumps(prefix,indent=2))
    result=json.loads(a.report.read_text()) if a.report.exists() else dict(status='WORKER_FAILED')
    result.update(exit_code=code,host_seconds=time.monotonic()-started);a.report.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    raise SystemExit(code)

if __name__=='__main__':main()
