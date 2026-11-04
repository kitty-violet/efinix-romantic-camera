"""Independently prove address projection and all remaining changed combinational boundaries."""
import argparse,copy,functools,json,pickle,re,subprocess,sys,time
from pathlib import Path
from run_units import ROOT,sha
from native_local_drivers import graph
from probe_full_native_split_map import params
from complete_native_logic_split import protected_blocks
from continuous_control_native import connections
from windows_process_tree_safe import stop_launched_tree
V=ROOT/'tmp/board-test/video-service-review1';sys.path.insert(0,str(V));from blif_control import write
p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);a=p.parse_args();N=a.native.resolve();start=time.monotonic()
r=json.loads((N/'joint-control-edit.json').read_text());P=Path(r['parent']);O=Path(r['plan_root']);plan=json.loads((O/'plan.json').read_text())
assert r['source_sha256']==sha(N/'native_source.v') and r['parent_sha256']==sha(P/'native_source.v')
for f,h in r['inputs'].items():assert sha(f)==h
key,old,op=pickle.loads((P/'zero-source-parse.pkl').read_bytes());assert key==r['parent_sha256']
text=(N/'native_source.v').read_text();parent=(P/'native_source.v').read_text();cache=N/'independent-joint-parse.pkl';parsekey=(sha(N/'native_source.v'),sha(Path(__file__).with_name('native_local_drivers.py')),sha(Path(__file__).with_name('probe_full_native_split_map.py')))
if cache.exists():ck,actual,pa=pickle.loads(cache.read_bytes());assert ck==parsekey
else:actual=graph(text);pa=params(text);cache.write_bytes(pickle.dumps((parsekey,actual,pa)))
g=copy.deepcopy(actual);adr=plan['address'];owned=set(adr['owned']);dr,loads=connections(old);newdr,newloads=connections(g)
zero=('literal',"1'b0",None);one=('literal',"1'b1",None)
ffnames=adr['source_ffs'];template=ffnames[0];q_to_d={old[n][1]['Q'][0]:old[n][1]['D'][0] for n in ffnames};ren={tuple(v['before']):tuple(v['after']) for v in adr['rename']}
assert len(ren)==len(owned) and set(ren)=={old[n][1]['O'][0] for n in owned}
for n in ffnames:
 assert actual[n]==old[n] and pa[n]==op[n]==op[template]
 assert {p:v for p,v in old[n][1].items() if p not in ('D','Q')}=={p:v for p,v in old[template][1].items() if p not in ('D','Q')}
assert op[template]['SR_SYNC']==(1,1) and op[template]['SR_SYNC_PRIORITY']==(1,1) and op[template]['SR_VALUE']==(1,0) and op[template]['D_POLARITY']==(1,1)
for n in owned:
 assert old[n][0]=='EFX_LUT4' and pa[n]==op[n]
 expected={p:[ren[t] if p=='O' else ren.get(t,q_to_d.get(t,t)) for t in ts] for p,ts in old[n][1].items()}
 assert actual[n]==('EFX_LUT4',expected),n
 # Every old input must be another moved gate, the original address Q, or a literal.
 assert all(t in ren or t in q_to_d or t in (zero,one) for p,ts in old[n][1].items() if p!='O' for t in ts)
@functools.lru_cache(None)
def reset_value(t):
 if t in q_to_d or t==zero:return 0
 if t==one:return 1
 n=dr[t];assert n in owned
 return op[n]['LUTMASK'][1]>>sum(reset_value(old[n][1].get('I'+str(i),[zero])[0])<<i for i in range(4))&1
projection_ffs=set()
for row in adr['projection']:
 n=row['old_output_cell'];ff=row['new_ff'];q=old[n][1]['O'][0];projection_ffs.add(ff)
 assert row['reset_value']==reset_value(q)
 assert actual[ff]==('EFX_FF',{**old[template][1],'D':[ren[q]],'Q':[q]})
 assert pa[ff]=={**op[template],'SR_VALUE':(1,reset_value(q))}
 g.pop(ff)
for t in ren.values():assert all(n in owned|projection_ffs for n,p in newloads[t]),('new address cone leaked',t)
for n in owned:g[n]=copy.deepcopy(old[n])
# This normalization is justified by induction: identical capture/hold/reset controls,
# D=f(address_D), Q=f(address_Q), reset=f(0). No input assumptions beyond defined controls.
assert {n for n in actual if actual[n][0]=='EFX_FF'}-{n for n in old if old[n][0]=='EFX_FF'}==projection_ffs
for n in old:
 if old[n][0] not in ('EFX_LUT4','EFX_ADD'):assert n in g and g[n]==old[n] and pa[n]==op[n],n
changed={n for n in old.keys()&g.keys() if old[n]!=g[n] or op[n]!=pa[n]};oc=changed|(old.keys()-g.keys());nc=changed|(g.keys()-old.keys())
assert all(old[n][0]=='EFX_LUT4' for n in oc) and all(g[n][0]=='EFX_LUT4' for n in nc)
assert protected_blocks(text)==protected_blocks(parent)
for pat in (r'^\s*(?:input|output|inout)\b.*$',r'^\s*assign\b.*$'):assert re.findall(pat,text,re.M)==re.findall(pat,parent,re.M)
for f in P.glob('*.sv'):assert sha(f)==sha(N/f.name)
ck,external,_=pickle.loads((O/'boundary.pkl').read_bytes());assert ck==key
outputs=sorted({old[n][1]['O'][0] for n in oc if old[n][1]['O'][0] in external or any(s not in oc for s,p in loads[old[n][1]['O'][0]])},key=str)
def cone(gr,targets):
 drivers,_=connections(gr);selected=set();stack=list(targets)
 while stack:
  t=stack.pop();n=drivers.get(t)
  if n is None or n in selected or gr[n][0] not in ('EFX_LUT4','EFX_ADD'):continue
  selected.add(n);stack.extend(t for p,ts in gr[n][1].items() if p not in ('O','CO') for t in ts)
 return selected
oc=cone(old,outputs);nc=cone(g,outputs);nd,nl=connections(g)
ob={t for n in oc for p,ts in old[n][1].items() if p not in ('O','CO') for t in ts if t[0]=='net' and dr.get(t) not in oc};nb={t for n in nc for p,ts in g[n][1].items() if p not in ('O','CO') for t in ts if t[0]=='net' and nd.get(t) not in nc}
for t in nb-ob:
 n=dr.get(t);assert n not in oc and n==nd.get(t)
 if n:assert old[n]==g[n] and op[n]==pa[n]
boundary=sorted(ob|nb,key=str)
tokens=set(boundary)|{t for gr,sel in ((old,oc),(g,nc)) for n in sel for p,ts in gr[n][1].items() if p in ('O','CO') for t in ts};names={t:'n'+str(i) for i,t in enumerate(sorted(tokens,key=str))}
def model(gr,ps,sel):
 nodes={}
 for n in sorted(sel):
  kind,pins=gr[n];ts=[pins.get('I'+str(i),[zero])[0] for i in range(4)] if kind=='EFX_LUT4' else [pins[p][0] for p in ('I0','I1','CI')]
  ins=list(dict.fromkeys(t for t in ts if t[0]=='net'))
  for port in (('O',) if kind=='EFX_LUT4' else tuple(p for p in ('O','CO') if p in pins)):
   mask=0
   for b in range(1<<len(ins)):
    e={zero:0,one:1,**{t:b>>i&1 for i,t in enumerate(ins)}}
    val=(ps[n]['LUTMASK'][1]>>sum(e[t]<<i for i,t in enumerate(ts))&1) if kind=='EFX_LUT4' else ((((e[ts[0]]^(1-ps[n]['I0_POLARITY'][1]))+(e[ts[1]]^(1-ps[n]['I1_POLARITY'][1]))+e[ts[2]])>>(1 if port=='CO' else 0))&1)
    mask|=val<<b
   nodes[names[pins[port][0]]]=dict(inputs=[names[t] for t in ins],mask=mask)
 return dict(inputs=[names[t] for t in boundary],outputs=[names[t] for t in outputs],nodes=nodes)
before=model(old,op,oc);after=model(g,pa,nc);write(N/'joint-original.blif',before);write(N/'joint-new.blif',after)
mutant=copy.deepcopy(after);q=mutant['nodes'][after['outputs'][0]];q['mask']^=(1<<(1<<len(q['inputs'])))-1;write(N/'joint-tampered.blif',mutant)
for label,right,want in [('equivalence','joint-new.blif','Networks are equivalent.'),('negative','joint-tampered.blif','NOT EQUIVALENT')]:
 s=N/(label+'.abc');s.write_text('cec joint-original.blif '+right+'\n')
 with (N/(label+'.log')).open('x') as log:
  child=subprocess.Popen([sys.executable,str(Path(__file__).with_name('run_installed_abc.py')),'--script',str(s)],cwd=N,stdout=log,stderr=subprocess.STDOUT)
  try:rc=child.wait(timeout=180)
  except subprocess.TimeoutExpired:(N/(label+'-stop.json')).write_text(json.dumps(stop_launched_tree(child)));raise
 body=(N/(label+'.log')).read_text();assert rc==0 and want in body,(label,rc,body)
 if label=='equivalence':assert 'NOT EQUIVALENT' not in body
cut=dict(old_cells=sorted(oc),new_cells=sorted(nc),outputs=outputs,boundary=boundary,tokens={v:k for k,v in names.items()})
(N/'joint-cut.json').write_text(json.dumps(cut,indent=2));(N/'joint-proof-graph.pkl').write_bytes(pickle.dumps((key,old,op,g,pa,cut)))
inputs={str(f):sha(f) for f in [Path(__file__),P/'native_source.v',P/'native-graph-audit.json',P/'weight-zero-independent-audit.json',N/'native_source.v',N/'joint-control-edit.json',N/'graft.json',O/'plan.json',O/'boundary.pkl',N/'joint-original.blif',N/'joint-new.blif',N/'joint-cut.json',N/'equivalence.log',N/'negative.log',V/'current-functional-shared-frontier-1.json']}
dest=N/'joint-control-independent-audit.json';proof=dict(status='JOINT_CONTROL_COMPOSITIONAL_CEC_PASS_NOT_PHYSICAL',source_sha256=sha(N/'native_source.v'),parent_sha256=key,primitive_count=len(actual),old_cut_primitives=len(oc),new_cut_primitives=len(nc),all_boundary_inputs=len(boundary),all_exposed_outputs=len(outputs),all_boolean_inputs=True,tampered_output_rejected=True,address_projection_ffs=len(projection_ffs),address_projection_induction=True,all_original_sequential_state_exact=True,ports_protected_assigns_exact=True,extra_cycles=0,arbitrary_X_equivalence=False,inputs=inputs,seconds=time.monotonic()-start,hardware_access=False,board_release=False)
dest.write_text(json.dumps(proof,indent=2));(N/'zero-source-parse.pkl').write_bytes(pickle.dumps((sha(N/'native_source.v'),actual,pa)))
wp=json.loads((P/'weight-zero-independent-audit.json').read_text());wp.update(source_sha256=sha(N/'native_source.v'),joint_control_proof=str(dest),scope='Exact address capture projection induction plus full Boolean changed-cone CEC; original state/IP/ports preserved; current vendor/map/interface tests required.',inputs={**wp['inputs'],**inputs,str(dest):sha(dest)});(N/'weight-zero-independent-audit.json').write_text(json.dumps(wp,indent=2))
na=json.loads((P/'native-graph-audit.json').read_text());na.update(retained_primitives=len(actual),joint_control_scope=proof['status'],inputs={**na['inputs'],**inputs,str(dest):sha(dest)});(N/'native-graph-audit.json').write_text(json.dumps(na,indent=2));print(json.dumps({k:v for k,v in proof.items() if k!='inputs'}),flush=True)
