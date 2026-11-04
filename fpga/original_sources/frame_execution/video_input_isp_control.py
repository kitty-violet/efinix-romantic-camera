"""Camera-input LUT protocol. Pure planning/verification, no USB imports."""
import math,zlib

def table(gamma=2.2,gains=(1.,1.,1.),black=0):
 if not isinstance(gamma,(int,float)) or not math.isfinite(gamma) or not 1<=gamma<=3:raise ValueError('gamma 1..3')
 if len(gains)!=3 or any(not math.isfinite(v) or not .25<=v<=4 for v in gains):raise ValueError('three gains .25..4')
 if type(black) is not int or not 0<=black<=64:raise ValueError('black 0..64')
 return bytes(int(255*(min(255,max(0,x-black)*gain)/255)**(1/gamma)+.5) for gain in gains for x in range(256))

def check_word(value):
 if type(value) is not int or not 0<=value<=0xffffffff:raise ValueError('ISP word range')
 if value in (0x49400000,0x49400001):return
 if value>>18==0x1240 and (value>>8&1023)<768:return
 if value>>16 in (0x4980,0x49c0):return
 if value>>8==0x4a0000:return
 raise ValueError('ISP opcode/address/reserved bits')

def upload_plan(data,swap=True):
 if not isinstance(data,bytes) or len(data)!=768:raise ValueError('exactly 3 x 256 table bytes')
 if type(swap) is not bool:raise ValueError('swap must be boolean')
 crc=zlib.crc32(data)
 words=[0x49400000|int(swap)]+[0x49000000|i<<8|b for i,b in enumerate(data)]+[0x49800000|(crc&65535),0x49c00000|(crc>>16)]
 for w in words:check_word(w)
 return dict(words=words,crc32=f'{crc:08x}',readback_rgb=[[data[i],data[256+i],data[512+i]] for i in range(256)],swap_rb=swap)

def verified(plan,status,readbacks,rejected_before):
 if not status.get('valid') or status.get('updating') or status.get('read_pending') or status.get('read_request'):raise ValueError('ISP not stable and valid')
 if status['count']!=768 or status['accepted_crc32']!=plan['crc32'] or status['swap_rb']!=plan['swap_rb']:raise ValueError('ISP accepted identity mismatch')
 if status['rejected']!=rejected_before:raise ValueError('ISP rejected a command')
 if readbacks!=plan['readback_rgb']:raise ValueError('ISP RAM readback mismatch')
 return True
