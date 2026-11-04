"""Supervised STOP-only camera-input calibration with complete RAM readback.

No download, Flash, DDR data writes, sensor writes or CNN start. Default only
validates a released bundle and prints the exact table identity, without USB.
"""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
from run_units import ROOT,sha
from video_input_isp_control import table,upload_plan,verified
from video_service_board import bound,ControlBus
from video_service_observation import fresh
from fixed_jtag_loader import MB,APB,snapshot
from windows_process_tree_safe import stop_launched_tree



def prepared_guard(bus):
 # Calibration runs after LOAD and STOP. The unloaded scratch guard is invalid here.
 s=snapshot(bus)
 checks={'mailbox':bus.read(MB),'pipeline':bus.read(APB+12),'ddr':bus.read(APB+0x6c)}
 if s['apb']['00']!=0x54493630:raise ValueError('Wrong identity')
 if s['phase']!=5 or s['apb']['08'] or s['apb']['60'] or checks['mailbox']:
  raise ValueError('Prepared stopped service required')
 if s['apb']['04']&24 or not s['apb']['04']&1 or (s['apb']['04']>>8)&4095!=2560:
  raise ValueError('Resident package not verified')
 if s['observed_coefficient_crc']!=0x7c6149bf or s['apb']['64']!=0xa008:
  raise ValueError('Different resident A identity')
 if checks['pipeline']!=0xb1000000 or checks['ddr']&255!=255 or checks['ddr']>>31&1:
  raise ValueError('Pipeline error, DDR unqualified, reset/PLL absent, or active CNN')
 return dict(snapshot=s,**checks)


def verified_tests(path):
 if path is None:raise ValueError('Regression receipt required')
 r=json.loads(path.read_text(encoding='utf-8'))
 if r.get('status')!='PASS' or r.get('tests',0)<12 or r.get('helper_sha256')!=sha(__file__) or r.get('test_sha256')!=sha(Path(__file__).with_name('test_video_service_input_isp_prepared.py')):
  raise ValueError('Tests do not bind this calibration helper')
 return r

def decode_status(words):
 if len(words)!=4:raise ValueError('Four ISP words required')
 c,w,r,crc=words
 if w&0x8000:raise ValueError('ISP reserved status bit')
 return dict(accepted_crc32=f'{crc:08x}',write_crc_internal=f'{c:08x}',count=w&1023,
  valid=bool(w&1024),updating=bool(w&2048),swap_rb=bool(w&4096),
  read_pending=bool(w&8192),read_request=bool(w&16384),rejected=w>>16,
  read_index=r>>24,read_rgb=[r>>(8*k)&255 for k in range(3)])


def configure(bus,plan,report,save,*,check_stopped,read_status):
 check_stopped();before=read_status(None)
 if before['rejected']==65535:raise ValueError('Saturated rejection counter')
 report.update(before=before,write_attempts=[],readbacks=[],status='UPLOADING_INPUT_TABLE');save()
 for i,word in enumerate(plan['words']):
  if i%64==0:check_stopped()
  report['write_attempts'].append(word);save()
  bus.write(MB+0x28,word)  # submit once; a failed acknowledgement never replays
  if bus.read(MB+0x28)!=word:raise IOError('ISP mailbox readback; no replay')
 check_stopped();report['accepted']=read_status(None);save()
 s=report['accepted']
 if not s['valid'] or s['updating'] or s['count']!=768 or s['accepted_crc32']!=plan['crc32'] or s['rejected']!=before['rejected']:
  raise ValueError('Hardware rejected input table; do not start')
 for index in range(256):
  if index%16==0:check_stopped()
  word=0x4a000000|index;report['write_attempts'].append(word);save()
  bus.write(MB+0x28,word)
  status=read_status(index)
  if status['read_index']!=index or status['read_pending'] or status['read_request'] or status['rejected']!=before['rejected']:
   raise ValueError('Stale/busy/rejected ISP read')
  report['readbacks'].append(status['read_rgb']);save()
 check_stopped();report['after']=read_status(None)
 verified(plan,report['after'],report['readbacks'],before['rejected'])
 report['status']='INPUT_ISP_CRC_AND_ALL_768_BYTES_VERIFIED_STOPPED';save()


def worker(a,plan):
 m,receipt=bound(a.manifest,a.program_receipt)
 if m['diagnostic_schema']!=6:raise ValueError('ISP schema6 required')
 out=dict(status='CONNECTING',manifest_sha256=sha(a.manifest),program_receipt_sha256=sha(a.program_receipt),
  helper_sha256=sha(__file__),test_receipt_sha256=sha(a.test_receipt),table_crc32=plan['crc32'],table_sha256=hashlib.sha256(a.data).hexdigest(),swap_rb=plan['swap_rb'],
  samples=[],isp_samples=[],hardware_access=False,flash_written=False,ddr_data_writes=0,cnn_started=False)
 def save():a.report.write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
 save();link=bus=None;deadline=time.monotonic()+570
 try:
  from fixed_jtag_physical_b import PhysicalBTransport
  from stable_jtag_reply import StableReplyTransport
  link=PhysicalBTransport(receipt['url']);out['hardware_access']=True
  bus=ControlBus(StableReplyTransport(link),writing=True,input_isp=True,seconds=570)
  out['guard']=prepared_guard(bus)
  def chunk():
   if time.monotonic()>=deadline:raise TimeoutError('Input calibration host deadline')
   return fresh(link,m['service_diagnostic_positions'],m['camera_diagnostic_slots'],[],lambda:None)
  def check_stopped():
   s=prepared_guard(bus)['snapshot'];out['samples'].append(s);save()
   if s['phase']!=5 or s['apb']['08'] or s['apb']['60'] or bus.read(MB):raise ValueError('Prepared STOP required; no automatic stop/restart')
   for _ in range(112):
    c=chunk()
    if c.get('diagnostic_profile')!='video-service-pages-v6':raise ValueError('Unexpected current schema')
    if c.get('page_kind')=='process' and c.get('chunk_index')==0 and c.get('measurement_valid'):
     v=c['chunk_words'][0]
     if not v&32 or v&0xcf:raise ValueError('Service not drained or has owner error')
     return
   raise TimeoutError('Missing fresh work-drained status')
  def read_status(index):
   words=[];last=None
   for _ in range(224):
    c=chunk()
    if c.get('diagnostic_profile')!='video-service-pages-v6':raise ValueError('Unexpected current schema')
    if c.get('page_kind')!='input_isp' or not c.get('measurement_valid'):continue
    part=c['chunk_index']
    if part==0:words=[];last=None
    if part!=len(words) or part>3:continue
    if last is not None and (c['page_sequence']-last)&255!=7:raise ValueError('Mixed ISP status sweep')
    words.append(c['chunk_words'][0]);last=c['page_sequence']
    if len(words)==4:
     value=decode_status(words)
     if index is not None and value['read_index']!=index:continue
     out['isp_samples'].append(value);save();return value
   raise TimeoutError('No fresh matching ISP readback; do not replay')
  configure(bus,plan,out,save,check_stopped=check_stopped,read_status=read_status)
 except Exception as e:out.update(status='FAIL_PREFIX_SAVED_NOT_ALLOWED_TO_START',error=repr(e));raise
 finally:
  if bus:out['control']=dict(transactions=bus.transactions,errors=bus.errors,attempts=bus.attempts)
  if link:link.close()
  save()


def main():
 p=argparse.ArgumentParser()
 for n in ['manifest','program-receipt','report']:p.add_argument('--'+n,type=Path,required=n=='manifest')
 p.add_argument('--test-receipt',type=Path)
 p.add_argument('--gamma',type=float,default=2.2);p.add_argument('--gains',type=float,nargs=3,default=[1,1,1]);p.add_argument('--black',type=int,default=0)
 p.add_argument('--swap-rb',choices=['yes','no'],default='yes')
 p.add_argument('--execute',action='store_true');p.add_argument('--supervised',action='store_true');p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
 a=p.parse_args();a.manifest=a.manifest.resolve();m=bound(a.manifest)
 if m['diagnostic_schema']!=6:raise ValueError('Current schema6 release required')
 a.data=table(a.gamma,a.gains,a.black);plan=upload_plan(a.data,a.swap_rb=='yes')
 if not a.execute:
  print(json.dumps(dict(status='INPUT_ISP_OFFLINE_VALIDATED_NO_USB',crc32=plan['crc32'],table_sha256=hashlib.sha256(a.data).hexdigest(),bytes=768,upload_writes=len(plan['words']))));return
 if not a.supervised or not a.program_receipt or not a.report:raise ValueError('Supervision, current receipt, new report required')
 verified_tests(a.test_receipt)
 a.program_receipt=a.program_receipt.resolve();a.report=a.report.resolve();bound(a.manifest,a.program_receipt)
 if not a.report.is_relative_to(ROOT/'tmp/board-test') or a.report.exists():raise ValueError('New project board-test report required')
 if a.worker:worker(a,plan);return
 cmd=[sys.executable,__file__,'--manifest',str(a.manifest),'--program-receipt',str(a.program_receipt),'--report',str(a.report),
  '--test-receipt',str(a.test_receipt.resolve()),'--gamma',str(a.gamma),'--gains',*map(str,a.gains),'--black',str(a.black),'--swap-rb',a.swap_rb,'--execute','--supervised','--worker']
 start=time.monotonic()
 with a.report.with_suffix('.log').open('x') as log:
  child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
  try:rc=child.wait(timeout=600);host=dict(exit_code=rc,seconds=time.monotonic()-start)
  except subprocess.TimeoutExpired:rc=124;host=dict(exit_code=rc,cleanup=stop_launched_tree(child))
 a.report.with_suffix('.host.json').write_text(json.dumps(host,indent=2));print(json.dumps(host));raise SystemExit(rc)
if __name__=='__main__':main()
