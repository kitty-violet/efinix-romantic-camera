"""Actual old/new map miter for health/path words at unchanged state outputs.

This complements the independent health/mode RTL tests and current header
oracle. It verifies changed selection/packing, not sequential-state behavior.
"""
import argparse,functools,json,pickle,random,re
from pathlib import Path
from run_units import ROOT,sha
from tap_project_status_binding import prove_status_apb
from tap_project_observation_binding import normalize_address_observation,normalize_tap_observation
from continuous_control_native import connections,net
from probe_full_native_split_map import params
from video_service_sample_ram import capture_roots as wide_roots
from video_service_narrow_sample_ram import capture_roots as narrow_roots
OLD_ALIAS='service_health_page[148]~FF'
NEW_ALIAS='service_health_page[457]~FF'
STATE_ALIASES={}
def state_name(n):return STATE_ALIASES.get(n,OLD_ALIAS if n==NEW_ALIAS else n)

def prove_front_bank_alias(old,new):
 a='service_front_bank[1]~FF';b='service_owner/disp_slot[1]~FF'
 assert a not in new['g'] and old['g'][a][0]==new['g'][b][0]=='EFX_FF'
 assert old['p'][a]==new['p'][b], 'duplicate state FF attributes differ'
 ap=old['g'][a][1];bp=new['g'][b][1]
 assert {k:v for k,v in ap.items() if k!='Q'}=={k:v for k,v in bp.items() if k!='Q'}, 'duplicate state input connections differ'
 assert set(ap)==set(bp)=={'D','CE','CLK','SR','Q'}
 assert old['p'][a]['SR_VALUE']==(1,0) and old['p'][a]['SR_SYNC']==(1,1)
 # Same clock, data, enable and reset with identical polarity/reset value:
 # both states agree after reset and inductively at every subsequent edge.
 STATE_ALIASES[b]=a
 return dict(old=a,new=b,all_non_output_pins_identical=True,all_parameters_identical=True,reset_inductive_state_alias=True)


def prove_merged_fault_bit(old,new):
 """Normalize the mapper's duplicate abort-bit choice at the health-state cut.

 Both instances are from byte-identical health RTL (separately tested).
 Their local D truth table, named control connections and FF attributes are
 compared exactly. This is not a proof of the complete sequential machine.
 """
 assert old['p'][OLD_ALIAS]==new['p'][NEW_ALIAS]
 for pin in ('CLK','CE','SR'):assert old['g'][OLD_ALIAS][1][pin]==new['g'][NEW_ALIAS][1][pin],pin
 a=old['d'][old['g'][OLD_ALIAS][1]['D'][0]];b=new['d'][new['g'][NEW_ALIAS][1]['D'][0]]
 assert old['g'][a][0]==new['g'][b][0]=='EFX_LUT4'
 assert {k:v for k,v in old['g'][a][1].items() if k!='O'}=={k:v for k,v in new['g'][b][1].items() if k!='O'}
 assert old['p'][a]==new['p'][b]
 source_old=old['path'].parent.parent/'video_service_health.sv';source_new=new['path'].parent.parent/'video_service_health.sv'
 assert sha(source_old)==sha(source_new)
 return dict(old=OLD_ALIAS,new=NEW_ALIAS,local_D_truth_cases=16,parameters_and_named_control_connections_identical=True,
  health_source_sha256=sha(source_new),complete_sequential_equivalence_claimed=False)

def prepare(board,roots_fn):
 path=board/'outflow/ti60f225_oob.map.v'
 key,g=pickle.loads((board/'service-validation-graph.pkl').read_bytes());assert key[0]==sha(path)
 ps=params(path.read_text());d,_=connections(g)
 return dict(path=path,g=g,p=ps,d=d,roots=roots_fn(g,ps))

def literal(text):
 m=re.fullmatch(r"\d+'[sS]?([hbd])([0-9a-fA-F]+)",text);assert m,text
 return int(m[2],{'h':16,'b':2,'d':10}[m[1]])

def evaluate(c,state,phase,part):
 g,d,ps=c['g'],c['d'],c['p'];leaves=set()
 apb_token=net('apb_enable')
 forced={net('service_page_phase',i):(phase>>i)&1 for i in range(3)}
 forced.update({net('service_page_part',i):(part>>i)&1 for i in range(4)})
 # Legacy selector state is deliberately a different implementation. Its
 # behavior is covered by the independent epoch proof; align the selector
 # values at this health/path combinational boundary.
 for i in range(2):
  value=state(('legacy_epoch',i));forced[net('service_legacy_epoch',i)]=value
  forced[net('service_page_sequence',i+3)]=value
 forced.update({net('service_pixel_part',i):0 for i in range(4)})
 @functools.lru_cache(None)
 def rank(t):
  if t in forced or t[0]=='literal':return 0
  n=d.get(t)
  if n is None or g[n][0] not in ('EFX_LUT4','EFX_ADD'):return 1000
  k,p=g[n]
  return 1+min((rank(q) for pin,ts in p.items() if pin not in ('O','CO') for q in ts),default=1000)
 @functools.lru_cache(None)
 def ev(t):
  if t==apb_token:
   leaves.add(('external',t));return state(('external',t))
  if t in forced:return forced[t]
  if t[0]=='literal':return literal(t[1])
  n=d.get(t)
  if n is None:
   leaves.add(('external',t));return state(('external',t))
  k,p=g[n]
  if k=='EFX_LUT4':
   lut=ps[n]['LUTMASK'][1];possible=list(range(16))
   inputs=[p.get('I'+str(i),[('literal',"1'b0",None)])[0] for i in range(4)]
   for i in sorted(range(4),key=lambda j:rank(inputs[j])):
    outputs={lut>>v&1 for v in possible}
    if len(outputs)==1:return outputs.pop()
    value=ev(inputs[i]);possible=[v for v in possible if v>>i&1==value]
   assert len(possible)==1
   return lut>>possible[0]&1
  if k=='EFX_ADD':
   value=sum(ev(p[z][0]) for z in ('I0','I1','CI'))
   return value&1 if p.get('O')==[t] else value>>1
  if k in ('EFX_RAM10','EFX_DPRAM10') and n.startswith('service_pixel_snapshot/storage/banks['):
   # Pixel data is another mux input. Vary it independently so a misplaced
   # phase/word selection cannot accidentally pass with zero pixel values.
   assert t[1]=='service_pixel_cache'
   leaves.add(('pixel_ram',t));return state(('pixel_ram',t))
  if k in ('EFX_RAM10','EFX_DPRAM10'):
   outputs=[(pin,ts.index(t)) for pin,ts in p.items() if pin in ('RDATA','DO','DOB','QA','QB','RDATAA','RDATAB') and t in ts]
   assert len(outputs)==1,(n,k,t,outputs)
   key=(n,*outputs[0]);leaves.add((k,key));return state((k,key))
  assert k=='EFX_FF',('unexpected status state',n,k,t)
  assert p['Q']==[t]
  leaves.add((k,state_name(n)));return state((k,state_name(n)))
 value=sum((ev(t)^invert)<<i for i,(t,invert) in c['roots'].items() if i<(128 if len(c['roots'])==192 else 32))
 return value,leaves

def main():
 p=argparse.ArgumentParser();p.add_argument('--old',type=Path,required=True);p.add_argument('--board',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 out=a.output.resolve();assert out.is_relative_to(ROOT/'tmp') and not out.exists()
 old=prepare(a.old.resolve(),wide_roots);new=prepare(a.board.resolve(),narrow_roots)
 merge=prove_merged_fault_bit(old,new)
 front_merge=prove_front_bank_alias(old,new)
 apb_binding=prove_status_apb(old,new)
 tap_binding=normalize_tap_observation(new)
 address_binding=normalize_address_observation(new)
 rng=random.Random(117803);checks=0;all_leaves=set()
 for trial in range(48):
  values={('external',net('apb_enable')):trial&1}
  def state(key):
   if key not in values:values[key]=rng.randrange(2)
   return values[key]
  for phase in (4,5):
   for part in range(16):
    wide,lw=evaluate(old,state,phase,part//4);narrow,ln=evaluate(new,state,phase,part)
    expected=(wide>>((part%4)*32))&0xffffffff
    if narrow!=expected:
     failure=dict(trial=trial,phase=phase,part=part,expected=hex(expected),actual=hex(narrow),different_bits=[i for i in range(32) if (expected^narrow)>>i&1],
       old_only=sorted(lw-ln,key=str),new_only=sorted(ln-lw,key=str),states=[(k,v) for k,v in values.items() if k in lw|ln])
     out.with_name('status-selection-miter-cut-failure.json').write_text(json.dumps(failure,indent=2))
     raise AssertionError((trial,phase,part,hex(expected),hex(narrow)))
    all_leaves|=lw|ln;checks+=1
 # All FF cuts are real common named state elements; no absent signal may
 # be silently accepted as zero. These states are not the changed transport.
 for kind,n in all_leaves:
  if kind=='pixel_ram':continue
  if kind in ('EFX_RAM10','EFX_DPRAM10'):
   name=n[0];assert old['g'][name][0]==new['g'][name][0]==kind,name
   assert old['p'][name]==new['p'][name],('different state RAM',name)
   continue
  if kind=='external':
   assert n in old['d'] or n in new['d'] or n[1] in ('i_pll_locked','pll_locked','process_pll_locked','user_pll_locked','DDR3_PLL_RSTN',
    'u_ddr3_top/controller/u_efx_ddr3_soft_controller/genblk1.inst_ddr3_axi/inst_efx_ddr3/top_mc/clk_rstn'),n
  else:
   new_name=next((alias for alias,original in STATE_ALIASES.items() if original==n),NEW_ALIAS if n==OLD_ALIAS else n)
   assert old['g'][n][0]==new['g'][new_name][0]=='EFX_FF',n
   assert not n.startswith(('service_serializer/','service_pixel_snapshot/')),n
 result=dict(status='ACTUAL_MAP_NARROW_HEALTH_PATH_SELECTION_PASS',cases=checks,bit_comparisons=checks*32,
  random_trials=48,state_cut_count=len(all_leaves),address_observation_binding=address_binding,tap_observation_binding=tap_binding,apb_observation_binding=apb_binding,merged_state_proof=merge,front_bank_alias_proof=front_merge,map_sha256=sha(new['path']),reference_map_sha256=sha(old['path']),
  inputs={**tap_binding['evidence'],**apb_binding['evidence'],**address_binding['evidence'],**{str(c['path']):sha(c['path']) for c in (old,new)},str(Path(__file__).with_name('validate_video_service_narrow_status_selection.py')):sha(Path(__file__).with_name('validate_video_service_narrow_status_selection.py')),str(Path(__file__)):sha(__file__)},checker_sha256=sha(__file__),
  scope='All16 words of health/path pages against prior128-bit actual-map selector, at common named state outputs. Original health/mode temporal behavior has separate RTL tests.',hardware_access=False,board_release=False)
 out.write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ('status','cases','bit_comparisons','state_cut_count')}))

if __name__=='__main__':main()
