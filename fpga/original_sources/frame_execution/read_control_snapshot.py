"""Bounded read-only diagnostics on USER1 selectors 2/3; optional one ID read."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from fixed_jtag_loader import APB,command_scan,decode_response
from windows_process_tree import stop_launched_tree
MAGIC=0x4f425331
LEVELS=('system_reset_n','key1_n','mipi_pll_locked','ddr_pll_locked','user_pll_locked','process_pll_locked','cal_done_raw','control_reset')
FLAGS=('control_reset','host_valid','host_ready','host_response','boot_valid','boot_ready','boot_failed','boot_terminal','fatal_bus_error','host_owner','apb_select','apb_enable','apb_ready','apb_error')

def decode(raw):
 if not isinstance(raw,int) or isinstance(raw,bool) or raw<0 or raw>=1<<258:raise ValueError('Invalid scan length/value')
 value=raw>>2
 if value&0xffffffff!=MAGIC or (value>>40)&255!=1 or (value>>36)&15:raise ValueError('OBS1 signature/version/reserved mismatch')
 valid=bool(value>>32&1);busy=bool(value>>33&1);ack=value>>34&1;request=value>>35&1
 if busy!=(request!=ack):raise ValueError('Inconsistent snapshot handshake')
 levels=value>>48&255
 out=dict(raw_hex=f'{value:064x}',valid=valid,busy=busy,sequence=value>>56&255,
          levels={name:bool(levels>>i&1) for i,name in enumerate(LEVELS)},payload=None)
 if valid:
  words=[value>>(64+32*i)&0xffffffff for i in range(6)];flags=words[4]
  if flags>>26:raise ValueError('Reserved payload bits set')
  out['payload']=dict(ticks=words[0],remote_bits=words[1]&255,remote_last=words[1]>>8&255,
     host_accepts=words[1]>>16&255,host_replies=words[1]>>24&255,
     last_host_address=f'{words[2]:08x}',last_host_reply=f'{words[3]:08x}',
     control_state=flags&15,flags={n:bool(flags>>(4+i)&1) for i,n in enumerate(FLAGS)},
     boot_fault=flags>>18&255,board_status=f'{words[5]:08x}')
 return out

class SnapshotPending(TimeoutError):pass

def fresh(transport,history,save,seconds=1.0):
 deadline=time.monotonic()+seconds
 def scan(selector):
  result=decode(transport.scan(selector,258));history.append(dict(selector=selector,**result));save();return result
 previous=scan(3)
 if previous['busy']:
  # A previous interrupted host request may still be in flight: drain it first.
  for _ in range(8):
   if time.monotonic()>=deadline:break
   previous=scan(3)
   if not previous['busy']:break
  if previous['busy']:raise SnapshotPending('System snapshot pending; TCK response retained')
 scan(2)
 for _ in range(8):
  if time.monotonic()>=deadline:break
  current=scan(3)
  if current['valid'] and not current['busy'] and current['sequence']==(previous['sequence']+1)&255:return current
 raise SnapshotPending('No fresh coherent snapshot; prefix retained')

def inspect(transport,result,save,probe_identity=False):
 result.update(idcode=f'{transport.idcode:08x}',usb_binding=transport.usb_binding,scans=[],snapshots=[])
 for _ in range(2):
  result['snapshots'].append(fresh(transport,result['scans'],save));save()
 first,last=result['snapshots'];result['system_ticks_advanced']=(last['payload']['ticks']-first['payload']['ticks'])&0xffffffff
 if not probe_identity:return
 if not result['system_ticks_advanced'] or last['levels']['control_reset'] or not last['levels']['system_reset_n']:
  result['identity_probe']='SKIPPED_CLOCK_OR_RESET';return
 # Exactly one read-only APB identity request; never issue Load/Run/DDR traffic.
 result['identity_requests']=1;save()
 packet,bits=command_scan(False,APB);transport.scan(packet,bits)
 try:
  deadline=time.monotonic()+2
  for _ in range(200):
   if time.monotonic()>=deadline:break
   response=decode_response(transport.scan(1,36))
   if response is not None:
    result['identity_probe']=f'{response:08x}'
    if response!=0x54493630:raise ValueError('APB identity mismatch')
    break
  else:raise TimeoutError('APB identity timeout')
  if 'identity_probe' not in result:raise TimeoutError('APB identity timeout')
 except Exception as exc:result['identity_error']=repr(exc)
 # A failed APB read does not prevent independently observing where it stopped.
 result['snapshots'].append(fresh(transport,result['scans'],save));save()

def worker(url,report,probe_identity=False):
 from fixed_jtag_physical_b import PhysicalBTransport
 result=dict(status='READING',url=url,read_only=True,hardware_access=True,
             identity_requests=0,ddr_accesses=0,fpga_programmed=False)
 def save():report.write_text(json.dumps(result,indent=2),encoding='utf-8')
 transport=None;save()
 try:
  transport=PhysicalBTransport(url);inspect(transport,result,save,probe_identity)
  result['status']='OBSERVATION_COMPLETE_NOT_BOARD_PASS'
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
 p.add_argument('--probe-identity',action='store_true');p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);a=p.parse_args()
 assert not a.report.exists();a.report.parent.mkdir(parents=True,exist_ok=True)
 if a.worker:
  result=worker(a.url,a.report,a.probe_identity);raise SystemExit(int(result['status']=='FAIL'))
 log=a.report.with_suffix('.log');assert not log.exists();started=time.monotonic()
 args=[sys._base_executable,str(Path(__file__).resolve()),'--worker','--url',a.url,'--report',str(a.report.resolve())]
 if a.probe_identity:args.append('--probe-identity')
 with log.open('w') as output:
  proc=subprocess.Popen(args,stdout=output,stderr=subprocess.STDOUT)
  try:code=proc.wait(timeout=20)
  except subprocess.TimeoutExpired:
   cleanup=stop_launched_tree(proc);code=124
   prefix=json.loads(a.report.read_text()) if a.report.exists() else {}
   prefix.update(status='HOST_TIMEOUT',cleanup=cleanup);a.report.write_text(json.dumps(prefix,indent=2))
 result=json.loads(a.report.read_text()) if a.report.exists() else dict(status='WORKER_FAILED')
 result.update(exit_code=code,host_seconds=round(time.monotonic()-started,3));a.report.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
 if code:raise SystemExit(code)
if __name__=='__main__':main()
