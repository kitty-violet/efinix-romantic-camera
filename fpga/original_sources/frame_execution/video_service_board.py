"""Supervised entry for the released single-button video service.

Default is offline validation. This entry never programs the FPGA or Flash.
An exact current temporary-JTAG receipt and explicit execute/supervised flags
are required before USB access. Only PREPARE writes the accepted A images.
"""
import argparse,json,subprocess,sys,time
from pathlib import Path
from run_units import ROOT,HERE,sha
from windows_process_tree_safe import stop_launched_tree
from fixed_jtag_loader import Bus,snapshot,validate_payload,MB,APB,command_scan,decode_response
from video_service_control import prepare_once,command_once,LOAD,LIVE,STOP,load_plan,diagnostic_word
from video_service_observation import fresh,prepared,path_status
from read_video_service_pages import PageAssembler
from read_camera_checked_snapshot_v2 import save_report

def check_diagnostic_word(value):
 if type(value) is not int or value>>16!=0x5653 or value&0xffff&~0x1f0d:raise ValueError('Diagnostic signature/reserved bits')
 if value&12 and not value&0x200:raise ValueError('Mode value without mode request')
 if value&0x800 and not value&0x400:raise ValueError('Result selector without inspection request')
 if value&0x400 and value&0x1000:raise ValueError('Conflicting inspection controls')

class ControlBus(Bus):
 def __init__(self,transport,*,writing=False,diagnostic=False,read_window=False,input_isp=False,seconds=120):
  super().__init__(transport,seconds);self.writing=writing;self.diagnostic=diagnostic;self.read_window=read_window;self.input_isp=input_isp;self.attempts=[]
 def access(self,writing,address,value=0):
  if type(address) is not int or address%4:raise ValueError('Aligned register address required')
  if writing and not self.writing:raise ValueError('Read-only observation session')
  allowed_write=address in (MB,MB+8,MB+12,MB+16,MB+20,MB+24)
  allowed_read=MB<=address<=MB+0x24 or address in {APB+x for x in (0,4,8,0xc,0x50,0x54,0x58,0x5c,0x60,0x64,0x68,0x6c)}
  if (self.diagnostic or self.input_isp) and address==MB+0x28:
   if writing:
    if self.input_isp:
     from video_input_isp_control import check_word
     check_word(value)
    else:check_diagnostic_word(value)
   allowed_write=allowed_read=True
  if self.read_window and 0xe1000000<=address<0xe1400000:allowed_read=True
  if not (allowed_write if writing else allowed_read):raise ValueError('Service register policy rejected address')
  if time.monotonic()>=self.deadline or len(self.attempts)>=4096:raise TimeoutError('Service control budget')
  packet,bits=command_scan(writing,address,value);self.attempts.append([writing,address,value])
  try:
   self.transport.scan(packet,bits) # send once; never retry a submitted write
   end=min(self.deadline,time.monotonic()+2)
   for _ in range(200):
    if time.monotonic()>=end:break
    reply=decode_response(self.transport.scan(1,36))
    if reply is not None:self.transactions+=1;return reply
   raise TimeoutError('No stable service reply')
  except Exception:self.errors+=1;raise

def diagnostic_once(bus,report,save,*,read_path,tx_pattern=None,mode=None,inspect=None,clear_first=False):
 if mode is not None or inspect is not None:
  state=snapshot(bus)
  if state['phase'] not in (1,5) or state['apb']['08'] or bus.read(MB):raise RuntimeError('Mode/inspection needs stopped, idle service')
 old=bus.read(MB+0x28)
 pattern=bool(old&1) if tx_pattern is None else tx_pattern
 value=diagnostic_word(tx_pattern=pattern,mode=mode,inspect=None if inspect=='clear' else inspect,clear_window=inspect=='clear',clear_first=clear_first)
 check_diagnostic_word(value)
 report.update(word=value,command_attempted=True,acknowledged=False);save()
 bus.write(MB+0x28,value)
 if bus.read(MB+0x28)!=value:raise IOError('Diagnostic mailbox readback mismatch; no replay')
 report['acknowledged']=True;report['path']=read_path();save()
 path=report['path']
 if mode is not None and path['mode']!=mode:raise RuntimeError('Hardware rejected mode change; slots must be empty')
 if inspect in ('input','result') and (not path['inspect_enabled'] or path['inspect_result']!=(inspect=='result') or path['inspect_bank']>=3):raise RuntimeError('Hardware did not lock a stopped front pair')
 if inspect=='clear' and path['inspect_enabled']:raise RuntimeError('Inspection window did not close')
 # TX request acknowledgement alone does not prove electrical HDMI or a
 # visible test pattern; the pixel-domain page and user's screen do that.
 return path

def read_window(bus,offset,size):
 if type(offset) is not int or type(size) is not int or offset<0 or size<=0 or size>4096 or offset%4 or size%4 or offset+size>0x400000:raise ValueError('Read at most 4096 aligned bytes within locked frame slot')
 return b''.join(bus.read(0xe1000000+i).to_bytes(4,'little') for i in range(offset,offset+size,4))

def bound(manifest,receipt=None):
 m=json.loads(manifest.read_text(encoding='utf8'))
 if m.get('status')!='VIDEO_SERVICE_OFFLINE_RELEASED_FOR_SUPERVISED_TEST':raise ValueError('Current service offline release required')
 if (m.get('diagnostic_schema'),m.get('service_diagnostic_positions')) not in ((4,list(range(192))),(5,list(range(96))),(6,list(range(96)))):raise ValueError('Wrong service diagnostic schema')
 if m.get('frequency_mhz')!=150 or m.get('flash_allowed') is not False:raise ValueError('Wrong deployment contract')
 if m.get('runtime_sha256',{}).get(str(Path(__file__).resolve()))!=sha(__file__):raise ValueError('Unbound runtime')
 for group in ('required_evidence','runtime_sha256'):
  if not m.get(group):raise ValueError('Missing release evidence')
  for f,h in m[group].items():
   if sha(f)!=h:raise ValueError('Changed release evidence: '+f)
 if sha(manifest.parent/'ti60f225_oob.bit')!=m['bitstream_sha256']:raise ValueError('Changed released bitstream')
 if receipt is not None:
  r=json.loads(receipt.read_text(encoding='utf8'))
  if r.get('status')!='TEMPORARY_JTAG_COMPLETE_PENDING_READ_ONLY_DIAGNOSTICS' or not r.get('fpga_programmed') or not r.get('user_supervised') or r.get('flash_written') is not False:raise ValueError('Supervised temporary-JTAG receipt required')
  if r['manifest_sha256']!=sha(manifest) or r['bitstream_sha256']!=m['bitstream_sha256']:raise ValueError('Different programmed release')
  return m,r
 return m

def worker(a):
 m,r=bound(a.manifest,a.program_receipt);report=a.report
 out=dict(status='CONNECTING',manifest_sha256=sha(a.manifest),program_receipt_sha256=sha(a.program_receipt),operation=a.operation,hardware_access=False,fpga_programmed=False,flash_written=False,cnn_started=False,scans=[],pages=[],register_samples=[])
 def save():save_report(report,out)
 save();link=None;stable=None;control=None
 try:
  from fixed_jtag_physical_b import PhysicalBTransport
  from stable_jtag_reply import StableReplyTransport
  link=PhysicalBTransport(r['url']);stable=StableReplyTransport(link)
  out.update(hardware_access=True,idcode=f'{link.idcode:08x}',usb_binding=link.usb_binding);save()
  control=ControlBus(stable,writing=a.operation!='observe',diagnostic=a.operation=='diagnose',seconds=1900 if a.operation=='prepare' else 120)
  positions=m['service_diagnostic_positions'];slots=m['camera_diagnostic_slots'];assembler=PageAssembler(slots)
  def observe(n=None):
   if n is None:n=(224 if m['diagnostic_schema']==6 else 192) if len(positions)==96 else 48
   for _ in range(n):
    c=fresh(link,positions,slots,out['scans'],save);page=assembler.add(c)
    if page is not None:out['pages'].append(page);save()
  def readiness():return prepared(link,positions,slots,out['scans'],save)
  def path_state():
   for _ in range((112 if m['diagnostic_schema']==6 else 96) if len(positions)==96 else 48):
    chunk=fresh(link,positions,slots,out['scans'],save)
    status=path_status(chunk)
    if status is not None:return status
   raise TimeoutError('No fresh atomic path status')
  observe();out['register_samples'].append(snapshot(control));save()
  if a.operation=='prepare':
   import ddr_scratch_bringup as scratch
   import accepted_a_ddr_stage as stage
   data=validate_payload(Path(m['a_integration']));load_plan(data)
   for name,b in data.items():
    if (a.manifest.parent/(name+'.bin')).read_bytes()!=b:raise ValueError('Bundle differs from accepted A: '+name)
   out['scratch']=dict(patterns=[],guards=[]);save()
   scratch.exercise(scratch.ScratchBus(stable),out['scratch'],save)
   out.update(status='UPLOADING_AND_READING_BACK_A',images=[],guards=[]);save()
   stage.transfer(stage.StageBus(stable,seconds=1740),data,out,save)
   if len(out['images'])!=3 or any(x['status']!='PASS' for x in out['images']):raise ValueError('Incomplete full readback')
   out['load']={};prepare_once(control,data,out['load'],save,read_prepared=readiness)
   out.update(status='READY_USE_SW0_TO_RUN_AND_STOP',cnn_started=False)
  elif a.operation in ('live','stop'):
   out['command']={};command_once(control,LIVE if a.operation=='live' else STOP,out['command'],save,read_prepared=readiness)
   out.update(status='LIVE_REQUEST_ACCEPTED_OBSERVE_SCREEN' if a.operation=='live' else 'STOP_REQUEST_ACCEPTED_DRAIN_PENDING',cnn_started=a.operation=='live')
  elif a.operation=='diagnose':
   out['diagnostic']={}
   path=diagnostic_once(control,out['diagnostic'],save,read_path=path_state,tx_pattern=None if a.tx_pattern is None else a.tx_pattern=='on',mode=a.mode,inspect=a.inspect,clear_first=a.clear_first)
   if a.read_bytes:
    if a.inspect not in ('input','result'):raise ValueError('Readout requires an explicitly locked input/result')
    control.read_window=True
    data=read_window(control,a.read_offset,a.read_bytes)
    after=path_state()
    if any(path[k]!=after[k] for k in ('mode','inspect_enabled','inspect_result','inspect_bank')):raise RuntimeError('Frame window changed during readout')
    file=report.with_suffix('.frame-prefix.bin');file.write_bytes(data)
    out['frame_prefix']=dict(file=str(file),sha256=sha(file),offset=a.read_offset,bytes=len(data),before=path,after=after,complete_frame=False)
   out['status']='DIAGNOSTIC_REQUEST_VERIFIED_OBSERVE_DISPLAY_PAGE'
  else:out['status']='READ_ONLY_SERVICE_OBSERVED_NOT_BOARD_PASS'
  observe();out['register_samples'].append(snapshot(control));save()
 except Exception as exc:out.update(status='FAIL_PREFIX_SAVED',error=repr(exc));save();return 1
 finally:
  if control:out['control']=dict(transactions=control.transactions,errors=control.errors,attempts=control.attempts)
  if stable:out.update(unstable_pairs=stable.unstable_pairs,reply_poll_pairs=stable.poll_pairs)
  if link:
   try:link.close()
   except Exception as exc:out.update(status='FAIL_PREFIX_SAVED',close_error=repr(exc))
  save()
 return int(out['status']=='FAIL_PREFIX_SAVED')

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--program-receipt',type=Path);p.add_argument('--report',type=Path)
 p.add_argument('--operation',choices=('observe','prepare','live','stop','diagnose'),default='observe')
 p.add_argument('--tx-pattern',choices=('on','off'));p.add_argument('--mode',type=int,choices=range(4));p.add_argument('--inspect',choices=('input','result','clear'));p.add_argument('--clear-first',action='store_true')
 p.add_argument('--read-offset',type=lambda s:int(s,0),default=0);p.add_argument('--read-bytes',type=int,default=0)
 p.add_argument('--execute',action='store_true');p.add_argument('--supervised',action='store_true');p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS);a=p.parse_args();a.manifest=a.manifest.resolve()
 if a.operation!='diagnose' and (a.tx_pattern is not None or a.mode is not None or a.inspect is not None or a.clear_first or a.read_bytes or a.read_offset):raise ValueError('Diagnostic options require diagnose operation')
 if a.read_bytes and (a.inspect not in ('input','result') or not 0<a.read_bytes<=4096 or a.read_bytes%4 or a.read_offset<0 or a.read_offset%4 or a.read_offset+a.read_bytes>0x400000):raise ValueError('Invalid bounded frame read')
 if not a.execute:bound(a.manifest);print('SERVICE_OFFLINE_BINDING_PASS_NO_USB');return
 if not a.supervised or not a.program_receipt or not a.report:raise ValueError('Onsite supervision, program receipt and new report required')
 a.report=a.report.resolve();a.program_receipt=a.program_receipt.resolve()
 if not a.report.is_relative_to(ROOT/'tmp/board-test') or a.report.exists():raise ValueError('Use a new project board-test report')
 if a.worker:raise SystemExit(worker(a))
 bound(a.manifest,a.program_receipt)
 budget=1920 if a.operation=='prepare' else 180
 cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--manifest',str(a.manifest),'--program-receipt',str(a.program_receipt),'--report',str(a.report),'--operation',a.operation,'--execute','--supervised','--worker']
 for flag,value in [('--tx-pattern',a.tx_pattern),('--mode',a.mode),('--inspect',a.inspect),('--read-offset',a.read_offset),('--read-bytes',a.read_bytes)]:
  if value is not None:cmd.extend([flag,str(value)])
 if a.clear_first:cmd.append('--clear-first')
 start=time.monotonic()
 with a.report.with_suffix('.log').open('x') as log:
  child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
  try:rc=child.wait(timeout=budget);host=dict(exit_code=rc,host_seconds=time.monotonic()-start)
  except subprocess.TimeoutExpired:host=dict(exit_code=124,host_seconds=time.monotonic()-start,cleanup=stop_launched_tree(child))
 save_report(a.report.with_suffix('.host.json'),host);print(json.dumps(host));raise SystemExit(host['exit_code'])
if __name__=='__main__':main()
