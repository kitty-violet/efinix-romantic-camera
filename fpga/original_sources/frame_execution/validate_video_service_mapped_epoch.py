"""Compare all128 nested legacy bits through actual mapped selector cones.

Cut only retained sequential/external producers before the old selector;
do not replace camera/CNN mux outputs with expected values.
"""
import argparse,functools,json,pickle,random,re
from pathlib import Path
from run_units import ROOT,sha
from video_service_graph import load
from video_service_serializer_native import P,net
from continuous_control_native import connections
from probe_full_native_split_map import params
from camera_joint_native import OUT
from video_service_sample_ram import capture_roots

def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--native',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 board=a.board.resolve();dst=a.out.resolve();assert dst.is_relative_to(ROOT/'tmp') and not dst.exists()
 src=board/'outflow/ti60f225_oob.map.v';key,g=pickle.loads((board/'service-validation-graph.pkl').read_bytes());assert key[0]==sha(src)
 ps=params(src.read_text());d,_=connections(g);D=load();og,op,od=D['g'],D['ps'],D['drivers']
 graft=json.loads((a.native/'graft.json').read_text());receipt=graft['serializer_extension'];assert receipt['selector_cone_only_observation']
 supplied={tuple(r['token']):r['expression'] for r in graft['supplied']}
 def literal(t):
  m=re.fullmatch(r"\d+'[sS]?([hbd])([\da-fA-F]+)",t);assert m,t
  return int(m[2],{'h':16,'b':2,'d':10}[m[1]])
 def mapped(t):
  if t[0]=='literal':return t
  if t in supplied:
   m=re.fullmatch(r'\\?([\w/]+)\s*(?:\[(\d+)\])?\s*',supplied[t]);assert m,(t,supplied[t])
   nt=net(m[1],int(m[2]) if m[2] is not None else None)
   # Resolve named native aliases through an actual retained selector pin.
   for n,pin in D['loads'][t]:
    if n in g and n not in graft['removed']:
     old_terms=og[n][1][pin];new_terms=g[n][1][pin]
     if len(old_terms)==len(new_terms):return new_terms[old_terms.index(t)]
   assert nt in d,(t,nt);return nt
  n=od.get(t)
  if n is None:return t
  kind,pins=og[n];assert n in g and g[n][0]==kind,(t,n)
  opts=[g[n][1][pin][ts.index(t)] for pin,ts in pins.items() if pin in OUT[kind] and t in ts]
  assert len(opts)==1,(t,n);return opts[0]
 raw=[]
 for row in receipt['payload'][832:960]:
  if row['token'] is None:raw.append((None,literal(row['expression'])));continue
  s=od[tuple(row['token'])];h=od[og[s][1]['D'][0]]
  assert s.startswith(P+'snapshot[') and h.startswith(P+'held_payload[')
  inv=row['invert']^(op[s]['D_POLARITY'][1]==0)^(op[h]['D_POLARITY'][1]==0)
  raw.append((og[h][1]['D'][0],int(inv)))
 boundaries=set();seen=set();todo=[t for t,inv in raw if t]
 while todo:
  t=todo.pop()
  if t in seen or t[0]=='literal':continue
  seen.add(t);n=od.get(t)
  if n is None or og[n][0] not in ('EFX_LUT4','EFX_ADD'):boundaries.add(t);continue
  kind,pins=og[n]
  todo.extend(q for pin,ts in pins.items() if pin not in OUT[kind] for q in ts)
 assert net(P+'request_sync') in boundaries and net('cnn_run_seen') in boundaries
 roots={i:r for i,r in capture_roots(g,ps).items() if i<128}
 assert len(roots)==128
 def evaluator(graph,parameters,drivers,values):
  @functools.lru_cache(None)
  def ev(t):
   if t in values:return values[t]
   if t[0]=='literal':return literal(t[1])
   n=drivers.get(t);assert n is not None,('uncut_external',t)
   kind,pins=graph[n]
   if kind=='EFX_LUT4':return parameters[n]['LUTMASK'][1]>>sum(ev(pins.get('I'+str(i),[('literal',"1'b0",None)])[0])<<i for i in range(4))&1
   if kind=='EFX_ADD':
    total=sum(ev(pins[z][0]) for z in ('I0','I1','CI'));return total&1 if pins.get('O')==[t] else total>>1
   raise AssertionError(('uncut_sequential',n,t))
  return ev
 rng=random.Random(246809);checks=0
 for trial in range(16):
  oldvals={t:rng.randrange(2) for t in sorted(boundaries,key=str)}
  # Inactive service-page inputs are explicit zero test stimuli, not
  # inferred values. Camera/CNN producers override these with random data.
  base={t:0 for n,(k,pins) in g.items() if k not in ('EFX_LUT4','EFX_ADD') for pin,ts in pins.items() if pin in OUT[k] for t in ts}
  base.update({t:0 for k,pins in g.values() for ts in pins.values() for t in ts if t[0]!='literal' and t not in d})
  base.update({mapped(t):v for t,v in oldvals.items() if t!=net('cnn_run_seen')})
  for seq in range(0,256,2):
   # Both source sample times must choose the same nested schema even when
   # the live CNN-start indicator or original request toggle changes.
   oldvals[net(P+'request_sync')]=seq>>3&1;oldvals[net('cnn_run_seen')]=seq>>4&1
   ev=evaluator(og,op,od,oldvals)
   expected=sum((inv if t is None else ev(t)^inv)<<i for i,(t,inv) in enumerate(raw))
   actual=0
   for chunk in (6,7):
    vals=base.copy()
    # Its old FF is now unconnected and trimmed; changing it cannot drive
    # the new selector. No fictitious net is injected into the mapped DUT.
    vals[mapped(net(P+'request_sync'))]=chunk&1
    for name,w,value in [('service_page_phase',3,chunk&1),('service_page_part',2,3),('service_page_sequence',8,seq+chunk-6)]:
     for i in range(w):vals[net(name,i)]=(value>>i)&1
    ev=evaluator(g,ps,d,vals)
    start=64 if chunk==6 else 0
    actual|=sum((ev(roots[i][0])^roots[i][1])<<(i-start) for i in range(start,start+64))<<(64*(chunk-6))
   assert actual==expected,(trial,seq,hex(actual),hex(expected));checks+=1
 result=dict(status='ACTUAL_MAP_NESTED_CAMERA_EPOCH_PASS',pairs=checks,compared_bits=checks*128,source_boundary_bits=len(boundaries),full_selector_cones=True,cnn_start_and_request_toggle_independent=True,map_sha256=sha(src),graft_sha256=sha(a.native/'graft.json'),checker_sha256=sha(__file__),hardware_access=False,board_release=False)
 dst.write_text(json.dumps(result,indent=2));print(result['status'],checks)
if __name__=='__main__':main()
