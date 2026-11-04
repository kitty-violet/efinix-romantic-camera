"""Bound read-only camera/OFP5 snapshot. Defaults to offline verification."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from run_units import sha
from read_camera_checked import decode
from read_control_snapshot import SnapshotPending
from windows_process_tree import stop_launched_tree

def checked_manifest(path):
 m=json.loads(path.read_text());assert m['status']=='OFFLINE_READY_AWAIT_USER_SUPERVISION'
 assert m['observation_format']=='OFP5' and m['camera_checks']['held_bits']==95
 assert len(m['camera_diagnostic_slots'])==111
 for f,h in m['required_evidence'].items():assert sha(f)==h,('Evidence changed',f)
 for f,h in m['bundle_files'].items():assert sha(path.parent/f)==h,('Bundle changed',f)
 assert sha(m['bitstream'])==m['bitstream_sha256'];return m

def fresh(transport,slots,history,save):
 deadline=time.monotonic()+1
 def scan(sel):
  d=decode(transport.scan(sel,1026),slots);history.append(dict(selector=sel,**d));save();return d
 old=scan(3)
 for _ in range(8):
  if not old['busy'] or time.monotonic()>deadline:break
  old=scan(3)
 if old['busy']:raise SnapshotPending('Previous OFP5 snapshot pending')
 scan(2)
 for _ in range(8):
  if time.monotonic()>deadline:break
  now=scan(3)
  if now['valid'] and not now['busy'] and now['sequence']==((old['sequence']+1)&255):return now
 raise SnapshotPending('No fresh camera/OFP5 snapshot; prefix retained')

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--url');p.add_argument('--report',type=Path)
 p.add_argument('--execute',action='store_true');p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);a=p.parse_args()
 m=checked_manifest(a.manifest.resolve())
 if not a.execute:print('OFFLINE_CAMERA_READER_BINDING_VERIFIED_NO_HARDWARE_ACCESS');return
 assert a.url and a.report and not a.report.exists();a.report.parent.mkdir(parents=True,exist_ok=True)
 if a.worker:
  from fixed_jtag_physical_b import PhysicalBTransport
  r=dict(status='READING',manifest_sha256=sha(a.manifest),expected_bit_sha256=m['bitstream_sha256'],hardware_access=True,read_only=True,ddr_accesses=0,fpga_programmed=False,scans=[],snapshots=[])
  def save():a.report.write_text(json.dumps(r,indent=2),encoding='utf8')
  t=None;save()
  try:
   t=PhysicalBTransport(a.url);r.update(idcode=f'{t.idcode:08x}',usb_binding=t.usb_binding)
   for _ in range(3):r['snapshots'].append(fresh(t,m['camera_diagnostic_slots'],r['scans'],save));save();time.sleep(.1)
   r['status']='CAMERA_SNAPSHOT_OBSERVED_NOT_FULL_BOARD_PASS'
  except SnapshotPending as e:r.update(status='SNAPSHOT_PENDING_NOT_BOARD_PASS',error=str(e))
  except Exception as e:r.update(status='FAIL',error=repr(e))
  finally:
   if t:
    try:t.close()
    except Exception as e:r.update(status='FAIL',close_error=repr(e))
   save()
  raise SystemExit(int(r['status']=='FAIL'))
 log=a.report.with_suffix('.log');assert not log.exists();start=time.monotonic()
 with log.open('x') as f:
  proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker','--execute','--manifest',str(a.manifest.resolve()),'--url',a.url,'--report',str(a.report.resolve())],stdout=f,stderr=subprocess.STDOUT)
  try:code=proc.wait(timeout=25)
  except subprocess.TimeoutExpired:
   cleanup=stop_launched_tree(proc);code=124
   r=json.loads(a.report.read_text()) if a.report.exists() else {};r.update(status='HOST_TIMEOUT',cleanup=cleanup);a.report.write_text(json.dumps(r,indent=2))
 r=json.loads(a.report.read_text()) if a.report.exists() else dict(status='WORKER_FAILED')
 r.update(exit_code=code,host_seconds=round(time.monotonic()-start,3));a.report.write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
 if code:raise SystemExit(code)
if __name__=='__main__':main()
