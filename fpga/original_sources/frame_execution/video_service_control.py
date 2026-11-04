"""Bounded LOAD/LIVE/STOP operations for the new service, no USB entry point.

The supervised caller must bind the current released bitstream and complete
DDR upload/readback before using these helpers. LOAD never starts CNN. This
module does not download, write Flash, open devices or enable background jobs.
"""
import hashlib,time,zlib
from fixed_jtag_loader import MB,APB,WINDOWS,snapshot

LOAD,LIVE,STOP=0x4c4f4144,0x4c495645,0x53544f50
TAG=0xa008
COEFFICIENT_SHA='ceebe7b3187fd47fbf0924883d19a9a335923d9de378ae09d30f362d9b9d0523'
EXPECTED_CRC={'coefficients-padded':0x7c6149bf,'compute':0xc89fa72e,'dma':0xfc4cc6a5}

def load_plan(payload):
 if set(payload)!=set(WINDOWS):raise ValueError('Expected all three accepted images')
 for name,data in payload.items():
  if len(data)!=WINDOWS[name][1] or zlib.crc32(data)!=EXPECTED_CRC[name]:raise ValueError('Accepted image length/CRC: '+name)
 if hashlib.sha256(payload['coefficients-padded']).hexdigest()!=COEFFICIENT_SHA:raise ValueError('Accepted A SHA')
 return [(MB+8,TAG),(MB+12,EXPECTED_CRC['coefficients-padded']),(MB+16,EXPECTED_CRC['compute']),
         (MB+20,EXPECTED_CRC['dma']),(MB+24,1),(MB,LOAD)]

def healthy(sample):
 if sample['phase']&0x80000000 or sample['apb']['04']&24 or sample['apb']['60']:
  raise RuntimeError('Controller, coefficient store or pipeline fault')

def prepare_once(bus,payload,report,save,*,read_prepared,now=time.monotonic,sleep=time.sleep):
 # read_prepared must obtain a NEW completed process/part3 diagnostic chunk,
 # not reuse a cached whole page. Parameters invalidate the previous READY.
 plan=load_plan(payload);before=snapshot(bus);healthy(before)
 if before['phase'] not in (1,5) or before['apb']['08'] or bus.read(MB):raise RuntimeError('LOAD requires idle empty command mailbox')
 count=before['apb']['5c'];report.update(status='PREPARING',before=before,write_attempts=[],samples=[]);save()
 for address,value in plan[:-1]:
  report['write_attempts'].append([address,value]);save();bus.write(address,value)
  if bus.read(address)!=value:raise IOError('Parameter readback mismatch; LOAD not sent')
 if read_prepared() is not False:raise RuntimeError('Previous READY did not invalidate')
 report['write_attempts'].append(list(plan[-1]));save();bus.write(*plan[-1])
 begin=now()
 while now()-begin<30:
  sample=snapshot(bus);healthy(sample);report['samples'].append(sample);save()
  if sample['apb']['08'] or sample['apb']['5c']!=count:raise RuntimeError('LOAD unexpectedly started processing')
  if sample['phase']==5 and bus.read(MB)==0 and read_prepared() is True:
   values=sample['apb']
   if sample['observed_coefficient_crc']!=EXPECTED_CRC['coefficients-padded'] or not values['04']&1 or (values['04']>>8)&4095!=2560:
    raise RuntimeError('Resident length/verified/CRC mismatch')
   if values['64']!=TAG or values['50']!=EXPECTED_CRC['compute'] or values['54']!=EXPECTED_CRC['dma']:
    raise RuntimeError('Tag/program CRC mismatch')
   report.update(status='RESIDENT_PREPARED_NOT_RUNNING',cnn_started=False,button_start_available=True);save();return
  sleep(.05)
 raise TimeoutError('LOAD deadline; do not replay command')

def command_once(bus,command,report,save,*,read_prepared=None):
 if command not in (LIVE,STOP):raise ValueError('Only LIVE or STOP')
 before=snapshot(bus);healthy(before)
 if bus.read(MB):raise RuntimeError('Pending command; do not overwrite')
 if command==LIVE:
  if before['phase']!=5 or before['apb']['08']:raise RuntimeError('LIVE requires successful LOAD/STOP')
  if read_prepared is None or read_prepared() is not True:raise RuntimeError('LIVE requires fresh verified READY')
 else:
  if before['phase']!=4 or before['apb']['08']!=3:raise RuntimeError('STOP requires active run')
 report.update(command=command,before=before,command_attempted=True,command_acknowledged=False);save()
 bus.write(MB,command)  # exactly one submission; read timeout never resends it
 report['command_acknowledged']=True;save()

def diagnostic_word(*,tx_pattern=False,clear_first=False,mode=None,inspect=None,clear_window=False):
 if mode not in (None,0,1,2,3) or inspect not in (None,'input','result'):raise ValueError('Invalid service mode/window')
 if inspect is not None and clear_window:raise ValueError('Contradictory window controls')
 return (0x56530000|int(tx_pattern)|(int(clear_first)<<8)|((0x200|(mode<<2)) if mode is not None else 0)
         |((0x400|(int(inspect=='result')<<11)) if inspect is not None else 0)|(int(clear_window)<<12))
