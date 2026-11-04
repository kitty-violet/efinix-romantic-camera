"""Inventory every new FF data/enable cone on the actual map.

This identifies crossings; it does not invent synchronizer safety or replace
routed held-data delay checks and the existing retained CDC evidence.
"""
import argparse,collections,functools,json,pickle
from pathlib import Path
from run_units import ROOT,HERE,sha
from video_service_graph import load
from continuous_control_native import connections,net
from camera_joint_native import used_inputs
from probe_full_native_split_map import params
from video_service_sample_ram import banks,storage_structure,bit_tokens

def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--native',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 board=a.board.resolve();out=a.out.resolve();src=board/'outflow/ti60f225_oob.map.v'
 assert out.is_relative_to(ROOT/'tmp') and not out.exists()
 key,g=pickle.loads((board/'service-validation-graph.pkl').read_bytes());assert key[0]==sha(src)
 parent=load();removed=set(json.loads((a.native/'graft.json').read_text())['removed'])
 retained=set(parent['g'])-removed;added=set(g)-retained
 d,l=connections(g);ps=params(src.read_text());active=set()
 @functools.lru_cache(None)
 def sources(t):
  if t[0]=='literal':return frozenset()
  n=d.get(t)
  if n is None:return frozenset([('external',t[1],t[2])])
  k,pins=g[n]
  if k not in ('EFX_LUT4','EFX_ADD'):return frozenset([('cell',n,None)])
  assert n not in active,('combinational_loop',n)
  active.add(n);result=set()
  for ts in used_inputs(k,pins,ps.get(n,{})).values():
   for q in ts:result.update(sources(q))
  active.remove(n);return frozenset(result)
 crossing=[];externals=[];unclassified=[];ff=[]
 for n in sorted(added):
  k,pins=g[n]
  if k!='EFX_FF':continue
  clock=pins['CLK'];ff.append(n)
  for pin in ('D','CE'):
   for kind,s,b in sorted(set().union(*(sources(t) for t in pins[pin])),key=str):
    if kind=='external':externals.append(dict(end=n+'|'+pin,source=s,bit=b,clock=clock));continue
    sk,sp=g[s]
    # RDATA is launched by RCLK; a tied-off ROM WCLK is not a clock crossing.
    clock_pins=('RCLK',) if sk in ('EFX_RAM10','EFX_DPRAM10') else ('CLK','RCLK','WCLK','CLKA','CLKB')
    clks=[v for q,v in sp.items() if q in clock_pins]
    if not clks:unclassified.append(dict(start=s,end=n+'|'+pin,kind=sk));continue
    if any(c!=clock for c in clks):crossing.append(dict(start=s,end=n+'|'+pin,source_clocks=clks,destination_clock=clock))
 # Verify first-stage outputs feed exactly their second-stage D pin.
 pairs=[('service_clear_meta~FF','service_clear_sync~FF'),('service_pixel_snapshot/req_meta~FF','service_pixel_snapshot/req_sync~FF'),
  ('service_pixel_snapshot/ack_meta~FF','service_pixel_snapshot/ack_sync~FF'),('service_display_observer/pattern_meta~FF','service_display_observer/pattern_sync~FF'),
  ('service_display_recovery/pixel_meta~FF','service_display_recovery/pixel_sync~FF'),('service_button/key_meta~FF','service_button/key_sync~FF')]
 pairs += [(f'service_health/lock_meta[{i}]~FF',f'service_health_page[{449+i}]~FF') for i in range(4)]
 chains=[]
 for first,second in pairs:
  k,x=g[first];k2,y=g[second];assert k==k2=='EFX_FF' and x['CLK']==y['CLK'] and x['Q']==y['D']
  assert set(l[x['Q'][0]])=={(second,'D')},(first,l[x['Q'][0]])
  assert x['CE']==y['CE']==[('literal',"1'b1",None)] and x['SR']==y['SR']
  assert ps[first].get('CLK_POLARITY')==ps[second].get('CLK_POLARITY')==(1,1)
  chains.append(dict(first=first,second=second,clock=x['CLK'],source=x['D'],parameters={first:ps[first],second:ps[second]}))
 ram_cross=[];ram_external=[];ram_structure=[]
 if banks(g,'service_serializer'):
  ram_structure=storage_structure(g,ps)
  for item in ram_structure:
   n=item['cell'];pins=g[n][1]
   for pin in ('WDATA','WE','WADDREN','WADDR','RE','RADDREN','RADDR','RST'):
    destination=pins['WCLK' if pin.startswith('W') else 'RCLK']
    bits=bit_tokens(pins[pin])
    for bit,t in enumerate(reversed(bits)):
     endpoint=n+'|'+pin+(f'[{bit}]' if len(bits)>1 else '')
     for kind,s,b in sorted(sources(t),key=str):
      if kind=='external':ram_external.append(dict(end=endpoint,source=s,bit=b,clock=destination));continue
      sk,sp=g[s]
      clocks=[sp[q] for q in (('RCLK',) if sk in ('EFX_RAM10','EFX_DPRAM10') else ('CLK','RCLK','WCLK','CLKA','CLKB')) if q in sp]
      assert clocks,('unclassified RAM source',n,pin,s,sk)
      if any(c!=destination for c in clocks):ram_cross.append(dict(start=s,end=endpoint,source_clocks=clocks,destination_clock=destination))
 result=dict(status='VIDEO_SERVICE_NEW_CDC_INVENTORIED_NOT_RELEASE',map_sha256=sha(src),new_ffs=len(ff),
  cross_domain_cones=crossing,external_inputs=externals,unclassified=unclassified,two_stage_chains=chains,
  dualclock_sample_ram=ram_structure,ram_input_cross_domain_cones=ram_cross,ram_external_inputs=ram_external,
  remaining='Classify all actual crossings, then check routed first-stage and held-data paths; retained CDC must also pass.',
  hardware_access=False,board_release=False,checker_sha256=sha(__file__))
 out.write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ['status','new_ffs']},indent=2))
 print('crossings',len(crossing),'external',len(externals),'unclassified',len(unclassified),'chains',len(chains))

if __name__=='__main__':main()
