"""Independent complete-graph same-edge induction and zero-consumer deletion proof."""
import argparse,copy,functools,json,pickle,re,time
from pathlib import Path
from run_units import ROOT,sha
from native_local_drivers import graph
from probe_full_native_split_map import params
from continuous_control_native import connections
from complete_native_logic_split import protected_blocks
p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);a=p.parse_args();N=a.native.resolve();start=time.monotonic()
r=json.loads((N/'tap-project-edit.json').read_text());P=Path(r['parent']);O=Path(r['plan_root']);plan=json.loads((O/'plan.json').read_text());ad=plan['address']
for f,h in r['inputs'].items():assert sha(f)==h,f
key,old,op=pickle.loads((P/'zero-source-parse.pkl').read_bytes());assert key==r['parent_sha256']==sha(P/'native_source.v')
text=(N/'native_source.v').read_text();parent=(P/'native_source.v').read_text();g=graph(text);pa=params(text);assert sha(N/'native_source.v')==r['source_sha256']
dr,loads=connections(old);nd,nl=connections(g);ck,external,ba=pickle.loads((O/'boundary.pkl').read_bytes());assert ck==key
ffnames=ad['source_ffs'];template=ffnames[0];owned=set(ad['owned']);ren={tuple(v['before']):tuple(v['after']) for v in ad['rename']};qs={old[n][1]['Q'][0]:old[n][1]['D'][0] for n in ffnames};Z=('literal',"1'b0",None);ONE=('literal',"1'b1",None)
for n in ffnames:
 assert old[n]==g[n] and op[n]==pa[n]==op[template]
 assert {p:v for p,v in old[n][1].items() if p not in ('D','Q')}=={p:v for p,v in old[template][1].items() if p not in ('D','Q')}
assert op[template]['D_POLARITY']==(1,1) and op[template]['SR_VALUE']==(1,0) and op[template]['SR_SYNC_PRIORITY']==(1,1)
assert set(ren)=={old[n][1]['O'][0] for n in owned}
@functools.lru_cache(None)
def reset(t):
 if t in qs or t==Z:return 0
 if t==ONE:return 1
 n=dr[t];assert n in owned
 return op[n]['LUTMASK'][1]>>sum(reset(old[n][1].get('I'+str(i),[Z])[0])<<i for i in range(4))&1
def verify_projection(graph,pars):
 for n in owned:
  assert old[n][0]=='EFX_LUT4' and op[n]==pars[n]
  expected={p:[ren[t] if p=='O' else ren.get(t,qs.get(t,t)) for t in ts] for p,ts in old[n][1].items()}
  assert graph[n]==('EFX_LUT4',expected)
  assert all(t in ren or t in qs or t in (Z,ONE) for p,ts in old[n][1].items() if p!='O' for t in ts)
 for row in ad['projection']:
  n=row['old_output_cell'];f=row['new_ff'];q=old[n][1]['O'][0]
  assert row['reset_value']==reset(q)
  assert graph[f]==('EFX_FF',{**old[template][1],'D':[ren[q]],'Q':[q]})
  assert pars[f]=={**op[template],'SR_VALUE':(1,reset(q))}
verify_projection(g,pa)
f=ad['projection'][0]['new_ff'];bad=copy.deepcopy(g);bad[f][1]['D']=[Z]
try:verify_projection(bad,pa)
except AssertionError:tamper=True
else:raise AssertionError('mutated projection accepted')
newffs={x['new_ff'] for x in ad['projection']};assert set(g)-set(old)==newffs
for t in ren.values():assert all(n in owned|newffs for n,p in nl[t])
live=set(old);dead=[]
for round in plan['dead_lut_rounds']:
 for n in round:
  assert n in live and old[n][0]=='EFX_LUT4' and old[n][1]['O'][0] not in external
  assert not any(s in live for s,p in loads[old[n][1]['O'][0]]),n
 live.difference_update(round);dead.extend(round)
assert set(old)-set(g)==set(dead) and len(dead)==141
normalized=copy.deepcopy(g);npa=copy.deepcopy(pa)
for n in newffs:normalized.pop(n);npa.pop(n)
for n in owned:normalized[n]=old[n];npa[n]=op[n]
assert normalized=={n:v for n,v in old.items() if n not in dead}
assert npa=={n:v for n,v in op.items() if n not in dead}
assert protected_blocks(text)==protected_blocks(parent)
for pat in (r'^\s*(?:input|output|inout)\b.*$',r'^\s*assign\b.*$'):assert re.findall(pat,text,re.M)==re.findall(pat,parent,re.M)
for f in P.glob('*.sv'):assert sha(f)==sha(N/f.name)
parentproof=json.loads((P/'joint-control-independent-audit.json').read_text());assert parentproof['source_sha256']==key and parentproof['status']=='JOINT_CONTROL_COMPOSITIONAL_CEC_PASS_NOT_PHYSICAL'
history=ROOT/'tmp/board-test/video-service-review1/current-functional-joint-control-1.json';h=json.loads(history.read_text());assert h['status']=='CURRENT_SERVICE_FUNCTIONAL_RELEASE_INPUTS_CHECKED_NOT_PHYSICAL_RELEASE'
for f,hash in h['evidence'].items():assert sha(f)==hash,f
inputs={str(f):sha(f) for f in [Path(__file__),P/'native_source.v',P/'native-graph-audit.json',P/'weight-zero-independent-audit.json',P/'joint-control-independent-audit.json',N/'native_source.v',N/'tap-project-edit.json',N/'graft.json',O/'plan.json',O/'boundary.pkl',history]}
proof=dict(status='TAP_PROJECT_COMPLETE_GRAPH_INDUCTION_PASS_NOT_PHYSICAL',source_sha256=sha(N/'native_source.v'),parent_sha256=key,primitive_count=len(g),projected_ffs=len(newffs),dead_luts=len(dead),projection_induction=True,all_other_graph_and_parameters_exact=True,all_original_state_exact=True,ports_protected_assigns_exact=True,tampered_D_rejected=tamper,extra_cycles=0,arbitrary_X_equivalence=False,inputs=inputs,seconds=time.monotonic()-start,hardware_access=False,board_release=False)
dest=N/'tap-project-independent-audit.json';dest.write_text(json.dumps(proof,indent=2));(N/'zero-source-parse.pkl').write_bytes(pickle.dumps((sha(N/'native_source.v'),g,pa)))
for name in ('weight-zero-independent-audit.json','native-graph-audit.json'):
 q=json.loads((P/name).read_text());q.update(source_sha256=sha(N/'native_source.v'),tap_project_proof=str(dest),inputs={**q['inputs'],**inputs,str(dest):sha(dest)})
 if name=='native-graph-audit.json':q['retained_primitives']=len(g)
 (N/name).write_text(json.dumps(q,indent=2))
print(json.dumps({k:v for k,v in proof.items() if k!='inputs'}),flush=True)
