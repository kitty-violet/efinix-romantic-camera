"""Actual mapped 192-bit capture input against independent legacy/page records.

The mapper trims 45 constant bits; their scan-time constants are checked by
the mapped serial test. Here every remaining capture FF D/polarity is checked.
"""
import argparse,functools,json,pickle,random,re
from pathlib import Path
from run_units import ROOT,sha
from continuous_control_native import connections,net
from probe_full_native_split_map import params
from video_service_graph import load
from video_service_serializer_native import P
from camera_joint_native import OUT
from video_service_sample_ram import capture_roots

def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--native',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 board=a.board.resolve();out=a.out.resolve();src=board/'outflow/ti60f225_oob.map.v'
 assert not out.exists() and out.is_relative_to(ROOT/'tmp')
 key,g=pickle.loads((board/'service-validation-graph.pkl').read_bytes());assert key[0]==sha(src)
 ps=params(src.read_text());d,l=connections(g);graft=json.loads((a.native/'graft.json').read_text());old=load()
 roots=capture_roots(g,ps)
 mask=sum(1<<i for i in roots);rng=random.Random(913214);checks=0;bit_checks=0
 values={}
 def setword(n,w,v):
  kept=0
  for i in range(w):
   t=net(n,i if w>1 else None);cell=d.get(t)
   if cell and g[cell][0] in ('EFX_FF','EFX_RAM10'):values[t]=(v>>i)&1;kept|=((v>>i)&1)<<i
  return kept
 supplied={tuple(r['token']):r['expression'] for r in graft['supplied']}
 def mapped(t):
  if t in supplied:
   m=re.fullmatch(r'\\?([\w/]+)\s*(?:\[(\d+)\])?\s*',supplied[t]);assert m,(t,supplied[t])
   nt=net(m[1],int(m[2]) if m[2] else None);assert nt in d,(t,nt)
   return nt
  n=old['drivers'].get(t)
  if n is None:return t
  k,pins=old['g'][n];assert g[n][0]==k
  options=[g[n][1][p][ts.index(t)] for p,ts in pins.items() if p in OUT[k] and t in ts]
  assert len(options)==1,(t,n,options)
  return options[0]
 def literal(t):
  m=re.fullmatch(r"\d+'[sS]?([hbd])([\da-fA-F]+)",t);assert m,t
  return int(m[2],{'h':16,'b':2,'d':10}[m[1]])
 def evaluate():
  @functools.lru_cache(None)
  def ev(t):
   if t in values:return values[t]
   if t[0]=='literal':return literal(t[1])
   cell=d.get(t)
   if cell is None:return 0
   kind,pins=g[cell]
   if kind in ('EFX_FF','EFX_RAM10','EFX_DPRAM10','EFX_SRL8','EFX_DSP24','EFX_DSP48'):return 0
   if kind=='EFX_LUT4':return ps[cell]['LUTMASK'][1]>>sum(ev(pins.get('I'+str(i),[('literal',"1'b0",None)])[0])<<i for i in range(4))&1
   if kind=='EFX_ADD':
    total=sum(ev(pins[z][0]) for z in ('I0','I1','CI'))
    return total&1 if pins.get('O')==[t] else total>>1
   raise AssertionError(('unexpected combinational root',cell,kind))
  return sum((ev(t)^invert)<<i for i,(t,invert) in roots.items())
 def check(expected,scope,checkmask=mask):
  nonlocal checks,bit_checks
  actual=evaluate();assert actual&checkmask==expected&checkmask,(scope,hex(actual),hex(expected),hex((actual^expected)&checkmask))
  checks+=1;bit_checks+=checkmask.bit_count()
 # Original 960-bit scan payload converted through actual original FF
 # polarities to its retained producer output. Randomize at exactly that cut.
 raw=[]
 for row in graft['serializer_extension']['payload']:
  if row['token'] is None:raw.append((None,literal(row['expression'])));continue
  s=old['drivers'][tuple(row['token'])];h=old['drivers'][old['g'][s][1]['D'][0]]
  assert s.startswith(P+'snapshot[') and h.startswith(P+'held_payload[')
  inv=row['invert']^(old['ps'][s]['D_POLARITY'][1]==0)^(old['ps'][h]['D_POLARITY'][1]==0)
  raw.append((mapped(old['g'][h][1]['D'][0]),int(inv)))
 assert len(raw)==960
 for trial in range(48):
  values.clear()
  for t,inv in raw:
   if t is not None and t[0]!='literal':values.setdefault(t,rng.randrange(2))
  legacy=sum(((inv if t is None else literal(t[1])^inv if t[0]=='literal' else values[t]^inv))<<i for i,(t,inv) in enumerate(raw))
  seq=rng.randrange(256);assert setword('service_page_sequence',8,seq)==seq
  for part in range(4):
   for phase in range(2):
    assert setword('service_page_phase',3,phase)==phase;assert setword('service_page_part',2,part)==part
    index=part*2+phase;expected=((legacy>>(128*index))&((1<<128)-1))|((0x4c4f0000|index)<<128)|(seq<<160)|(4<<168)
    check(expected,('legacy',trial,index))
 fields=['service_process_ticks','service_video_recoveries','service_capture_admitted','service_capture_skipped','service_capture_orphans','service_compute_admitted','committed_frames','service_repeated','dropped_results','service_pixel_sequence','service_pixel_timeouts','service_rejected_buttons']
 for trial in range(48):
  values.clear();actual={n:setword(n,32,rng.getrandbits(32)) for n in fields}
  for name in ('w_arstn','ceg_net4871','u_ddr3_top/controller/u_efx_ddr3_soft_controller/genblk1.inst_ddr3_axi/inst_efx_ddr3/top_mc/clk_rstn'):values[net(name)]=1
  setword('cal_done_sync',1,1);setword('service_prepared',1,1)
  pixel=rng.getrandbits(128);assert setword('service_pixel_cache',128,pixel)==pixel
  seq=rng.randrange(256);assert setword('service_page_sequence',8,seq)==seq
  flags=[rng.randrange(2) for _ in range(3)]
  for n,v in zip(['service_pixel_valid','service_pixel_busy','service_pixel_stale'],flags):
   physical=v^int(n=='service_pixel_stale');assert setword(n,1,physical)==physical
  for phase in range(2,6):
   for part in range(4):
    setword('service_page_phase',3,phase);setword('service_page_part',2,part);setword('service_pixel_part',2,part)
    head=((0x56530000|(part<<3)|phase)<<128)|(seq<<160)|(4<<168)|(flags[0]<<176)|(flags[1]<<177)|(flags[2]<<178)
    check(head,('service_header',trial,phase,part),mask&~((1<<128)-1))
    if phase==2:check(head|pixel,('pixel',trial,part))
    if phase==3:
     words=[0x53525631,actual[fields[0]],actual[fields[1]],1,*[actual[n] for n in fields[2:]],0,48];assert len(words)==16
     page=0
     for w in words:page=(page<<32)|w
     check(head|((page>>(128*part))&((1<<128)-1)),('process',trial,part))
 result=dict(status='ACTUAL_MAP_COMPACT_LEGACY_SERVICE_CAPTURE_PASS',binary_checks=checks,bit_comparisons=bit_checks,random_trials_per_group=48,capture_bits=sorted(roots),trimmed_constant_bits=192-len(roots),scope='All8 original legacy chunks; every service header; all pixel/process payload words. Health/path payload source tests separate. Trimmed constants and handshake verified by mapped serial test.',map_sha256=sha(src),native_graft_sha256=sha(a.native/'graft.json'),checker_sha256=sha(__file__),hardware_access=False,board_release=False)
 out.write_text(json.dumps(result,indent=2));print(result['status'],checks,bit_checks)
if __name__=='__main__':main()
