"""Exercise actual mapped LUT/ADD paths into retained JTAG capture flops.

Binary combinational boundary test: all service headers, pixel payload, and
process counters. This is separate from mailbox/transport temporal tests.
"""
import argparse,functools,json,pickle,random,re
from pathlib import Path
from run_units import ROOT,sha
from continuous_control_native import connections,net
from probe_full_native_split_map import params
from video_service_display_native import O

def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--native',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 board=a.board.resolve();out=a.out.resolve();src=board/'outflow/ti60f225_oob.map.v'
 assert not out.exists() and out.is_relative_to(ROOT/'tmp')
 key,g=pickle.loads((board/'service-validation-graph.pkl').read_bytes());assert key[0]==sha(src)
 ps=params(src.read_text());d,l=connections(g);graft=json.loads((a.native/'graft.json').read_text())
 # Positions are selected by the emitted graft; no assumption that the
 # packet occupies the low bits of the entire1024-bit JTAG transport.
 positions=sorted(int(re.search(r'held_payload\[(\d+)\]',n)[1]) for n in g if n.startswith(O+'held_payload['))[:183]
 assert len(positions)==183
 roots=[g[O+f'held_payload[{i}]~FF'][1]['D'][0] for i in positions]
 rng=random.Random(932178);checks=0
 fields=['service_process_ticks','service_video_recoveries','service_capture_admitted','service_capture_skipped','service_capture_orphans',
  'service_compute_admitted','committed_frames','service_repeated','dropped_results','service_pixel_sequence','service_pixel_timeouts','service_rejected_buttons']
 for trial in range(32):
  values={};actual={}
  def setword(n,w,v):
   kept=0
   for i in range(w):
    t=net(n,i if w>1 else None);cell=d.get(t)
    if cell and g[cell][0]=='EFX_FF':values[t]=(v>>i)&1;kept|=((v>>i)&1)<<i
    elif v>>i&1:pass # Trimmed constant-zero bits are not falsely randomized.
   return kept
  for n in fields:actual[n]=setword(n,32,rng.getrandbits(32))
  for name in ('w_arstn','ceg_net4871','u_ddr3_top/controller/u_efx_ddr3_soft_controller/genblk1.inst_ddr3_axi/inst_efx_ddr3/top_mc/clk_rstn'):
   values[net(name)]=1
  setword('cal_done_sync',1,1);setword('service_prepared',1,1)
  pixel=rng.getrandbits(128);assert setword('service_pixel_cache',128,pixel)==pixel
  seq=rng.randrange(256);assert setword('service_page_sequence',8,seq)==seq
  flags=[rng.randrange(2) for _ in range(3)]
  for n,v in zip(['service_pixel_valid','service_pixel_busy','service_pixel_stale'],flags):
   # stale initializes1 in RTL. The mapper stores its complement in a
   # zero-initialized FF and inverts it at consumers (LUT truth checked here).
   physical=v^int(n=='service_pixel_stale')
   assert setword(n,1,physical)==physical
  for phase in range(2,6):
   for part in range(4):
    setword('service_page_phase',3,phase);setword('service_page_part',2,part);setword('service_pixel_part',2,part)
    @functools.lru_cache(None)
    def ev(t):
     if t in values:return values[t]
     if t[0]=='literal':
      m=re.fullmatch(r"\d+'[sS]?([hbd])([\da-fA-F]+)",t[1]);assert m,t
      return int(m[2],{'h':16,'b':2,'d':10}[m[1]])
     cell=d.get(t)
     if cell is None:return 0
     kind,pins=g[cell]
     # This is a combinational cut: un-driven state outputs (including
     # unrelated legacy RAM/DSP paths behind the phase mux) are fixed zero.
     if kind in ('EFX_FF','EFX_RAM10','EFX_DPRAM10','EFX_SRL8','EFX_DSP24','EFX_DSP48'):return 0
     if kind=='EFX_LUT4':return ps[cell]['LUTMASK'][1]>>sum(ev(pins.get('I'+str(i),[('literal',"1'b0",None)])[0])<<i for i in range(4))&1
     if kind=='EFX_ADD':
      total=sum(ev(pins[z][0]) for z in ('I0','I1','CI'))
      return total&1 if pins.get('O')==[t] else total>>1
     raise AssertionError(('unexpected combinational root',cell,kind))
    packet=sum(ev(t)<<i for i,t in enumerate(roots))
    header=(0x56530000|(part<<3)|phase)|(seq<<32)|(2<<40)|(flags[0]<<48)|(flags[1]<<49)|(flags[2]<<50)
    assert packet>>128==header,('header',trial,phase,part,hex(packet>>128),hex(header))
    checks+=1
    if phase==2:assert packet&((1<<128)-1)==pixel,('pixel',trial,part);checks+=1
    if phase==3:
     words=[0x53525631,actual[fields[0]],actual[fields[1]],1,*[actual[n] for n in fields[2:]],0,48]
     assert len(words)==16
     page=0
     for w in words:page=(page<<32)|w
     expected=(page>>(128*part))&((1<<128)-1)
     assert packet&((1<<128)-1)==expected,('process',trial,part,hex(packet&((1<<128)-1)),hex(expected));checks+=1
 result=dict(status='ACTUAL_MAP_JTAG_SERVICE_HEADER_PIXEL_PROCESS_PAYLOAD_PASS',binary_checks=checks,random_trials=32,
  field_scope='All four service phase headers and every part; full128-bit pixel cache and all16 process words. Health/path payload temporal and source tests remain separate.',
  map_sha256=sha(src),native_graft_sha256=sha(a.native/'graft.json'),capture_positions=positions,checker_sha256=sha(__file__),hardware_access=False,board_release=False)
 out.write_text(json.dumps(result,indent=2));print(result['status'],checks)

if __name__=='__main__':main()
