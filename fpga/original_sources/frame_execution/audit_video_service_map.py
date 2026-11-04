"""Bind retained native primitives through official synthesis; diagnose deltas."""
import argparse,collections,json,re,time,traceback,pickle
from pathlib import Path
from run_units import ROOT,sha
from native_local_drivers import graph,render_token,lexical_tokens
from probe_full_native_split_map import params,lut_function
from complete_native_logic_split import protected_blocks
from probe_native_open_ports import compare_boundary
from audit_native_roundtrip import tokens
from audit_local_reset_implementation import norm

def audit(board):
 started=time.monotonic();source=(board/'native_source.v').read_text();mapped=(board/'outflow/ti60f225_oob.map.v').read_text()
 cache=board/'retained-audit-parse.pkl'
 key=[sha(board/'native_source.v'),sha(board/'outflow/ti60f225_oob.map.v'),sha(Path(__file__).with_name('native_local_drivers.py')),sha(Path(__file__).with_name('probe_full_native_split_map.py'))]
 if cache.exists():
  old,a,b,pa,pb=pickle.loads(cache.read_bytes());assert old==key
 else:
  a,b=graph(source),graph(mapped);pa,pb=params(source),params(mapped)
  cache.write_bytes(pickle.dumps((key,a,b,pa,pb),protocol=4))
 # This alias is emitted explicitly by the new single owner. It is not an
 # inferred equivalence between two unrelated signals.
 alias_old=('net','video_processing_pipeline/pipeline/capture/start_fire',None)
 alias_new=('net','camera_burst_capture/start_fire',None)
 assert re.search(r'assign\s+\\video_processing_pipeline/pipeline/capture/start_fire\s*=\s*\\camera_burst_capture/start_fire\s*;',source)
 for _,pins in a.values():
  for p,ts in pins.items():pins[p]=[alias_new if t==alias_old else t for t in ts]
 missing=sorted(a.keys()-b.keys());added=sorted(b.keys()-a.keys());assert not missing,('retained_missing',missing[:20])
 # The narrowed observation counter has a declared zero upper half. The
 # mapper may legally collapse its old diagnostic readback LUTs. Prove that
 # exact boundary, then compare each affected LUT's full Boolean function;
 # a pin-position bijection alone is invalid after constant propagation.
 folded=set();constant_boundary={}
 owner=board/'frame_pair_owner.sv'
 if owner.exists() and "assign dropped_results={16'd0,dropped_count};" in owner.read_text():
  assert re.search(r'reg\[15:0\] dropped_count,repeated_count;',owner.read_text())
  exported=re.search(r'frame_pair_owner service_owner\(.*?\.service_native_state\(\{([^}]+)\}\)',source,re.S)
  produced=re.search(r'assign service_native_state=\{([^}]+)\};',owner.read_text())
  assert exported and produced
  destination=[lexical_tokens(x.strip()) for x in exported[1].split(',')]
  origin=[re.sub(r'\s+','',x.strip().lstrip('\\')) for x in produced[1].split(',')]
  assert len(destination)==len(origin)
  for i in range(16,32):
   wanted=[('net','dropped_results',i)]
   assert destination[origin.index(f'dropped_results[{i}]')]==wanted
  constant_boundary={('net','dropped_results',i):('literal',"1'b0",None) for i in range(16,32)}
  for n,(kind,pins) in a.items():
   if any(t in constant_boundary for ts in pins.values() for t in ts):
    assert kind=='EFX_LUT4',(n,kind)
    folded.add(n)
    for p,ts in pins.items():pins[p]=[constant_boundary.get(t,t) for t in ts]
 forward={};reverse={};conflicts=[]
 def bind(x,y,where):
  if x[0]=='literal' or y[0]=='literal':
   if x!=y:conflicts.append((where,x,y,'literal'))
   return
  if x in forward and forward[x]!=y:conflicts.append((where,x,y,'forward',forward[x]));return
  if y in reverse and reverse[y]!=x:conflicts.append((where,x,y,'reverse',reverse[y]));return
  forward[x]=y;reverse[y]=x
 for n,(kind,pins) in a.items():
  assert b[n][0]==kind,(n,kind,b[n][0])
  if kind!='EFX_LUT4':assert pa.get(n,{})==pb.get(n,{}),('parameter',n,pa.get(n),pb.get(n))
  for p in pins.keys() & b[n][1].keys():
   if n in folded and p!='O':continue
   x,y=pins[p],b[n][1][p];assert len(x)==len(y),(n,p,x,y)
   for v,w in zip(x,y):bind(v,w,n+'.'+p)
 result=dict(status='RETAINED_MAP_GRAPH_INSPECTION',retained=len(a),added=len(added),added_kinds=dict(collections.Counter(b[n][0] for n in added)),
  constant_folded_luts=sorted(folded),constant_boundary_owner_sha256=sha(owner) if constant_boundary else None,
  conflicts=len(conflicts),examples=conflicts[:30],source_sha256=sha(board/'native_source.v'),map_sha256=sha(board/'outflow/ti60f225_oob.map.v'),hardware_access=False,board_release=False)
 (board/'retained-map-inspection.json').write_text(json.dumps(result,indent=2))
 assert not conflicts,('retained_net_bijection',len(conflicts),conflicts[:8])
 for n,(kind,pins) in a.items():
  if n in folded:
   for ts in pins.values():
    for t in ts:assert t[0]=='literal' or t in forward,('unbound folded LUT net',n,t)
  q={p:('' if not v else render_token(forward.get(v[0],v[0])) if len(v)==1 else '{'+', '.join(render_token(forward.get(t,t)) for t in v)+'}') for p,v in pins.items()}
  z={p:('' if not v else render_token(v[0]) if len(v)==1 else '{'+', '.join(map(render_token,v))+'}') for p,v in b[n][1].items()}
  if kind=='EFX_LUT4':assert lut_function(q,pa[n]['LUTMASK'][1])==lut_function(z,pb[n]['LUTMASK'][1]),('LUT_function',n)
  else:assert q==z,('ports',n,q,z)
 renamed={tokens(norm(render_token(x)))[0]:tokens(norm(render_token(y)))[0] for x,y in forward.items()}
 boundary=compare_boundary(source,mapped,renamed)
 parent=ROOT/'tmp/frame-execution/connected-board-continuous-srl-register1/native_source.v'
 assert protected_blocks(source)==protected_blocks(parent.read_text()),'encrypted input altered'
 assert not any(n.startswith('protected_inst/') for n in added),'unexpected public protected expansion'
 result.update(status='VIDEO_SERVICE_RETAINED_MAPPED_GRAPH_PASS',seconds=time.monotonic()-started,checker_sha256=sha(__file__),protected_boundary=boundary,protected_input_unchanged=True,protected_internal_equivalence='Official mapper re-encrypts output; internal equivalence not asserted by public graph checker',new_logic_equivalence='Not proved by retained-cell check; dedicated mapped boundary/CDC checks required')
 (board/'retained-map-audit.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);a=p.parse_args()
 print(json.dumps(audit(a.board.resolve()),indent=2))
