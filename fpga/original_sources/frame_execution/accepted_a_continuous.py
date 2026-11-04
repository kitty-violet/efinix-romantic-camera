"""Bounded continuous command only; callers must establish full upload/readback."""
import time
import fixed_jtag_loader as host
import accepted_a_ddr_stage as stage

TAG=0xa008
MAX_FRAMES=512
HOST_RUN_SECONDS=90
READS={host.APB+x for x in (0,4,8,0x50,0x54,0x58,0x5c,0x60,0x64)}
READS.update(host.MB+x for x in (0,4,8,12,16,20,24,28,32,36))

def plan(payload,frames):
 if type(frames) is not int or not 1<=frames<=MAX_FRAMES:raise ValueError('frame budget must be 1..512')
 return [(host.MB+8,TAG),(host.MB+12,int(stage.metadata(payload['coefficients-padded'])['crc32'],16)),
  (host.MB+16,int(stage.metadata(payload['compute'])['crc32'],16)),(host.MB+20,int(stage.metadata(payload['dma'])['crc32'],16)),
  (host.MB+24,frames),(host.MB,0x434e4e31)]

class RunBus(host.Bus):
 def __init__(self,transport,payload,frames):
  super().__init__(transport,seconds=HOST_RUN_SECONDS+10)
  self.plan=plan(payload,frames);self.frames=frames;self.armed=False;self.write_attempts=[];self.attempts=0
 def access(self,writing,address,value=0):
  if self.attempts>=20000:raise RuntimeError('bounded preview command count exceeded')
  if writing:
   index=len(self.write_attempts)
   if not self.armed or index>=len(self.plan) or (address,value)!=self.plan[index]:
    raise ValueError('only five parameters and one START; no automatic retry')
   self.write_attempts.append((address,value))
  elif address not in READS:raise ValueError('preview reads limited to status and mailbox')
  self.attempts+=1
  return super().access(writing,address,value)

def run_once(bus,payload,out,save,*,now=time.monotonic,sleep=time.sleep):
 before=host.snapshot(bus)
 if before['phase'] not in (1,5) or before['apb']['08'] or before['apb']['04']&24 or before['apb']['60']:
  raise RuntimeError('loader must be idle or successfully stopped, with no fault')
 out.update(before=before,samples=[],frame_budget=bus.frames);save()
 bus.armed=True
 for address,value in bus.plan[:-1]:
  bus.write(address,value)
  if bus.read(address)!=value:raise IOError('run parameter readback differs; START not sent')
 bus.write(*bus.plan[-1]);out['start_command_acknowledged']=True;save()
 begin=now();seen_new_phase=False
 while now()-begin<HOST_RUN_SECONDS:
  s=host.snapshot(bus);s['elapsed_host_s']=now()-begin
  count=(s['apb']['5c']-before['apb']['5c'])&0xffffffff
  s['committed_since_start']=count;out['samples'].append(s);save()
  if s['phase']&0x80000000 or s['apb']['04']&24 or s['apb']['60']:
   raise RuntimeError('boot/store/pipeline fault during preview')
  if count>bus.frames:raise RuntimeError('committed frames exceeded explicit budget')
  if s['phase']!=5:seen_new_phase=True
  # A second START can be acknowledged before the polling loader consumes it.
  # Do not mistake the previous successful phase=5/count=0 for a new completion.
  if s['phase']==5 and (seen_new_phase or count):
   if count!=bus.frames or s['apb']['08']!=0:raise RuntimeError('finite frame budget or STOP did not hold')
   if s['observed_coefficient_crc']!=bus.plan[1][1]:raise RuntimeError('coefficient CRC differs')
   if (s['apb']['04']>>8)&4095!=2560 or not s['apb']['04']&1:raise RuntimeError('resident image not verified')
   if s['apb']['64']!=TAG or s['apb']['50']!=bus.plan[2][1] or s['apb']['54']!=bus.plan[3][1]:raise RuntimeError('tag/program CRC differs')
   if s['apb']['58']<=0:raise RuntimeError('missing CNN cycle measurement')
   out.update(committed_frames=count,last_cnn_cycles=s['apb']['58'],last_cnn_ms_at_150mhz=s['apb']['58']/150000,
    automatic_stop_observed=True,screen_frame_validated=False,host_poll_times_are_not_hardware_frame_timestamps=True)
   return
  sleep(.05)
 raise TimeoutError('90s preview observation expired; never resend START')
