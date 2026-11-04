"""OFS1 first-fault readout. Selectors 2/3 only; no functional bus request."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from read_control_snapshot import LEVELS,FLAGS,SnapshotPending
from windows_process_tree import stop_launched_tree
MAGIC=0x4f465331
STATE_NAMES=('IDLE','DECODE','MAIL_READ','APB_SETUP','APB_ACCESS','AXI_WRITE','AXI_B','AXI_AR','AXI_R','RESPONSE')

def state(raw):
    return dict(raw=f'{raw:03x}',legal=raw!=0 and raw&(raw-1)==0,
                asserted=[name for i,name in enumerate(STATE_NAMES) if raw>>i&1])

def flags(raw):
    if raw>>26:raise ValueError('Reserved control flags set')
    return dict(compressed_state=raw&15,boot_fault=raw>>18&255,
                signals={name:bool(raw>>(4+i)&1) for i,name in enumerate(FLAGS)})

def decode(raw):
    if not isinstance(raw,int) or isinstance(raw,bool) or raw<0 or raw>=1<<258:raise ValueError('Invalid scan length/value')
    value=raw>>2
    if value&0xffffffff!=MAGIC or value>>40&255!=1 or value>>36&15:raise ValueError('OFS1 signature/version/reserved mismatch')
    valid=bool(value>>32&1);busy=bool(value>>33&1);ack=value>>34&1;request=value>>35&1
    if busy!=(request!=ack):raise ValueError('Inconsistent snapshot handshake')
    levels=value>>48&255
    result=dict(raw_hex=f'{value:064x}',valid=valid,busy=busy,sequence=value>>56&255,
                levels={n:bool(levels>>i&1) for i,n in enumerate(LEVELS)},payload=None)
    if valid:
        w=[value>>(64+32*i)&0xffffffff for i in range(6)];m=w[1];reason=m>>26&15
        result['payload']=dict(captured=bool(m>>31),armed=bool(m>>30&1),ticks_at_capture=w[0],
            reason={n:bool(reason>>i&1) for i,n in enumerate(('illegal_control_state','early_bus_stall_64_cycles','fatal_bus_error','boot_failed'))},
            boot_state=m>>20&63,control_state=state(m&1023),previous_control_state=state(m>>10&1023),
            address=f'{w[2]:08x}',flags=flags(w[3]),previous_flags=flags(w[4]),previous_address=f'{w[5]:08x}')
    return result

def fresh(transport,history,save,seconds=1):
    deadline=time.monotonic()+seconds
    def scan(selector):
        result=decode(transport.scan(selector,258));history.append(dict(selector=selector,**result));save();return result
    previous=scan(3)
    for _ in range(8):
        if not previous['busy'] or time.monotonic()>=deadline:break
        previous=scan(3)
    if previous['busy']:raise SnapshotPending('Snapshot request still pending')
    scan(2)
    for _ in range(8):
        if time.monotonic()>=deadline:break
        current=scan(3)
        if current['valid'] and not current['busy'] and current['sequence']==(previous['sequence']+1)&255:return current
    raise SnapshotPending('No fresh snapshot; valid prefix retained')

def worker(url,report):
    from fixed_jtag_physical_b import PhysicalBTransport
    result=dict(status='READING',url=url,hardware_access=True,read_only=True,identity_requests=0,ddr_accesses=0,
                fpga_programmed=False,scans=[],snapshots=[])
    def save():report.write_text(json.dumps(result,indent=2),encoding='utf-8')
    transport=None;save()
    try:
        transport=PhysicalBTransport(url);result.update(idcode=f'{transport.idcode:08x}',usb_binding=transport.usb_binding)
        for _ in range(2):result['snapshots'].append(fresh(transport,result['scans'],save));save()
        result['status']='FIRST_FAULT_OBSERVED_NOT_BOARD_PASS'
    except SnapshotPending as exc:result.update(status='SNAPSHOT_PENDING_NOT_BOARD_PASS',error=str(exc))
    except Exception as exc:result.update(status='FAIL',error=repr(exc))
    finally:
        if transport:
            try:transport.close()
            except Exception as exc:result.update(status='FAIL',close_error=repr(exc))
        save()
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);a=p.parse_args()
    assert not a.report.exists();a.report.parent.mkdir(parents=True,exist_ok=True)
    if a.worker:raise SystemExit(int(worker(a.url,a.report)['status']=='FAIL'))
    log=a.report.with_suffix('.log');assert not log.exists();started=time.monotonic()
    with log.open('x') as output:
        child=subprocess.Popen([sys._base_executable,str(Path(__file__).resolve()),'--worker','--url',a.url,'--report',str(a.report.resolve())],stdout=output,stderr=subprocess.STDOUT)
        try:code=child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            cleanup=stop_launched_tree(child);code=124
            prefix=json.loads(a.report.read_text()) if a.report.exists() else {}
            prefix.update(status='HOST_TIMEOUT',cleanup=cleanup);a.report.write_text(json.dumps(prefix,indent=2))
    result=json.loads(a.report.read_text()) if a.report.exists() else dict(status='WORKER_FAILED')
    result.update(exit_code=code,host_seconds=round(time.monotonic()-started,3))
    a.report.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    if code:raise SystemExit(code)
if __name__=='__main__':main()
