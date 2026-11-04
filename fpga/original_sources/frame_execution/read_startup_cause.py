"""Offline OFP5 decoder. Raw primitive samples are checked against fixed stimuli."""
import read_fault_probe as previous
MAGIC=0x4f465035
SCAN_BITS=1026
def cell_expected():
 q=0;raw=0;rows=[]
 for index in range(8):
  data=(0x52>>index)&1;disable=(0xd4>>index)&1;reset_n=(0xee>>index)&1
  if not reset_n:q=0
  elif not disable:q=data
  o=(0x1144>>(index&3))&1;p=(0x1144>>(8+(index&3)))&1
  sample=o|(p<<1)|(q<<2)|(q<<3)|(q<<4);raw|=sample<<(5*index)
  rows.append(dict(index=index,data=data,disable_ce=disable,reset_n=reset_n,expected=sample))
 return raw,rows
def cell_decode(v):
 expected,rows=cell_expected();raw=v&((1<<40)-1)
 valid=(v>>56==0xc5 and v>>55&1 and v>>54&1 and (v>>48&63)==32 and (v>>40&255)==255)
 samples=[(raw>>(5*i))&31 for i in range(8)]
 return dict(raw=f'{v:016x}',complete=bool(valid),samples=samples,expected=[r['expected'] for r in rows],
  matches=bool(valid and raw==expected),mismatch_patterns=[i for i,r in enumerate(rows) if samples[i]!=r['expected']],
  scope='Finite local primitive/control patterns, not a proof of all device resources or DDR calibration')
def decode(raw):
 if not isinstance(raw,int) or isinstance(raw,bool) or not 0<=raw<1<<SCAN_BITS:raise ValueError('OFP5 length/value')
 value=raw>>2
 if value&0xffffffff!=MAGIC or value>>40&255!=5:raise ValueError('OFP5 signature/version')
 base=value&((1<<768)-1)
 base=(base&~(0xffffffff|(255<<40)))|previous.MAGIC|(4<<40)
 result=previous.decode(base<<2);result.update(format='OFP5',raw_hex=f'{value:0256x}')
 if result['valid']:
  active=value>>768&((1<<64)-1);bad=value>>832&((1<<64)-1)
  result['startup_trace']=dict(first_active=dict(valid=bool(active>>63),previous_cycle_valid=bool(active>>62&1),
   seen_handshake_mask=active>>56&15,tick=active>>32&0xffffff,live=f'{active&0xffffffff:08x}'),
   first_unsupported_transition=dict(valid=bool(bad>>63),first_after_reset=bool(bad>>62&1),
    read_state=bad>>56&7,write_state=bad>>59&7,tick=bad>>32&0xffffff,preceding_live=f'{bad&0xffffffff:08x}'))
  result['cell_probes']={name:cell_decode(value>>(896+64*i)&((1<<64)-1)) for i,name in enumerate(('process150','ddr400'))}
 return result
