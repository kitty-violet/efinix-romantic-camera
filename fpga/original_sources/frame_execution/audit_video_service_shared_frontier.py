"""Independent emitted-source delta/whole-boundary CEC; no timing or hardware claims."""
import argparse,copy,json,pickle,re,subprocess,sys,time
from pathlib import Path
from run_units import ROOT,sha
from native_local_drivers import graph
from probe_full_native_split_map import params
from complete_native_logic_split import protected_blocks
from continuous_control_native import connections
from windows_process_tree_safe import stop_launched_tree
V=ROOT/'tmp/board-test/video-service-review1';sys.path.insert(0,str(V));from blif_control import write
p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);a=p.parse_args();N=a.native.resolve();start=time.monotonic()
r=json.loads((N/'shared-frontier-edit.json').read_text());P=Path(r['parent']);O=ROOT/'tmp/frame-execution/video-service63-batch-screen1'
assert r['source_sha256']==sha(N/'native_source.v') and r['parent_sha256']==sha(P/'native_source.v')
for f,h in r['inputs'].items():assert sha(f)==h
key,old,op=pickle.loads((P/'zero-source-parse.pkl').read_bytes());assert key==r['parent_sha256']
text=(N/'native_source.v').read_text();parent=(P/'native_source.v').read_text();cache=N/'independent-frontier-parse.pkl';parsekey=(sha(N/'native_source.v'),sha(Path(__file__).with_name('native_local_drivers.py')),sha(Path(__file__).with_name('probe_full_native_split_map.py')))
if cache.exists():
 ck,actual,pa=pickle.loads(cache.read_bytes());assert ck==parsekey
else:
 actual=graph(text);pa=params(text);cache.write_bytes(pickle.dumps((parsekey,actual,pa)))
g=copy.deepcopy(actual)
plan=json.loads((O/'plan.json').read_text());copies=next(e['rows'] for e in plan['edits'] if e['kind']=='physical_copies');alias={tuple(c['new_output']):tuple(c['old_output']) for c in copies}
fusion_survey=json.loads((O/'fusion-survey.json').read_text());fusion_result=json.loads((ROOT/'tmp/frame-execution/video-service63-fusions1/result.json').read_text())
assert set(r['accepted_fusions'])<=set(fusion_result['accepted']) and fusion_result['survey_sha256']==sha(O/'fusion-survey.json')
def root(t):
 seen=set()
 while t in alias:assert t not in seen;seen.add(t);t=alias[t]
 return t
for c in copies:
 n=c['copy'];base=c['original'];assert g[n][0]==g[base][0]=='EFX_LUT4' and pa[n]==pa[base]
 assert {p:[root(t) for t in ts] for p,ts in g[n][1].items()}==g[base][1],n
 expected=set(map(tuple,c['sinks']))
 for i in r['accepted_fusions']:
  row=fusion_survey['rows'][i];expected={x for x in expected if x[0] not in (row['parent'],row['target'])}
  expected|={(sn,p) for sn,(k,pins) in row['new_graph'].items() for p,ts in pins.items() if p!='O' and tuple(c['new_output']) in list(map(tuple,ts))}
 actuals={(sn,p) for sn,(k,pins) in g.items() for p,ts in pins.items() if p!='O' and tuple(c['new_output']) in ts};assert actuals==expected,(n,actuals^expected)
 g.pop(n)
for k,pins in g.values():
 for pin,ts in pins.items():
  if pin!='O':pins[pin]=[root(t) for t in ts]
changed={n for n in old.keys()&g.keys() if old[n]!=g[n] or op[n]!=pa[n]};oc=changed|(old.keys()-g.keys());nc=changed|(g.keys()-old.keys())
assert all(old[n][0] in ('EFX_LUT4','EFX_ADD') for n in oc) and all(g[n][0]=='EFX_LUT4' for n in nc)
aux={row['removed'] for e in plan['edits'] if e['kind']=='four_writer_overflow' for row in e['rows']}
assert {n for n in oc if old[n][0]=='EFX_ADD'}==aux
# Include actual carry producers, never treat a carry and its source bits as independent inputs.
carryparents={n for e in plan['edits'] if e['kind']=='four_writer_overflow' for row in e['rows'] for n in row['old_graph']}-aux
assert all(old[n]==g[n] and op[n]==pa[n] for n in carryparents)
oc|=carryparents;nc|=carryparents
for n in old.keys()&g.keys()-changed:assert old[n]==g[n] and op[n]==pa[n]
assert protected_blocks(text)==protected_blocks(parent)
for pat in (r'^\s*(?:input|output|inout)\b.*$',r'^\s*assign\b.*$'):assert re.findall(pat,text,re.M)==re.findall(pat,parent,re.M)
for f in P.glob('*.sv'):assert sha(f)==sha(N/f.name)
_,external,_=pickle.loads((O/'boundary.pkl').read_bytes());dr,loads=connections(old)
outputs=sorted({t for n in oc for p,ts in old[n][1].items() if p in ('O','CO') for t in ts if t in external or any(s not in oc for s,p in loads[t])},key=str)
# A union of edited nodes is not a closed cut: unchanged gates can correlate
# two of its apparent leaves. Expand BOTH real cones to state/IP boundaries.
def complete_cone(gr,targets):
 drivers,_=connections(gr);selected=set();stack=list(targets)
 while stack:
  t=stack.pop();n=drivers.get(t)
  if n is None or n in selected or gr[n][0] not in ('EFX_LUT4','EFX_ADD'):continue
  selected.add(n);stack.extend(t for p,ts in gr[n][1].items() if p not in ('O','CO') for t in ts)
 return selected
oc=complete_cone(old,outputs);nc=complete_cone(g,outputs)
boundary=sorted({t for n in oc for p,ts in old[n][1].items() if p not in ('O','CO') for t in ts if t[0]=='net' and dr.get(t) not in oc},key=str)
nd,nl=connections(g);newboundary={t for n in nc for p,ts in g[n][1].items() if p not in ('O','CO') for t in ts if t[0]=='net' and nd.get(t) not in nc}
boundary_extension=newboundary-set(boundary)
for t in boundary_extension:
 # An extra independent leaf strengthens CEC: the old output must agree for
 # every value, including values the newly factored function may cancel.
 producer=dr.get(t);assert producer not in oc and producer==nd.get(t),(t,producer)
 if producer is not None:assert old[producer]==g[producer] and op[producer]==pa[producer]
boundary=sorted(set(boundary)|newboundary,key=str)
assert all(nd.get(t) in nc for t in outputs)
tokens=set(boundary)|{t for gr,sel in ((old,oc),(g,nc)) for n in sel for p,ts in gr[n][1].items() if p in ('O','CO') for t in ts};names={t:'n'+str(i) for i,t in enumerate(sorted(tokens,key=str))};ZERO=('literal',"1'b0",None);ONE=('literal',"1'b1",None)
def model(gr,ps,sel):
 nodes={}
 for n in sorted(sel):
  kind,pins=gr[n];ts=[pins.get('I'+str(i),[ZERO])[0] for i in range(4)] if kind=='EFX_LUT4' else [pins[p][0] for p in ('I0','I1','CI')]
  if kind=='EFX_ADD':assert set(ps[n])=={'I0_POLARITY','I1_POLARITY'} and all(v in ((1,0),(1,1)) for v in ps[n].values())
  ins=list(dict.fromkeys(t for t in ts if t[0]=='net'))
  for port in (('O',) if kind=='EFX_LUT4' else tuple(p for p in ('O','CO') if p in pins)):
   mask=0
   for b in range(1<<len(ins)):
    env={ZERO:0,ONE:1,**{t:b>>i&1 for i,t in enumerate(ins)}}
    value=(ps[n]['LUTMASK'][1]>>sum(env[t]<<i for i,t in enumerate(ts))&1) if kind=='EFX_LUT4' else ((((env[ts[0]]^(1-ps[n]['I0_POLARITY'][1]))+(env[ts[1]]^(1-ps[n]['I1_POLARITY'][1]))+env[ts[2]])>>(1 if port=='CO' else 0))&1)
    mask|=value<<b
   nodes[names[pins[port][0]]]=dict(inputs=[names[t] for t in ins],mask=mask)
 return dict(inputs=[names[t] for t in boundary],outputs=[names[t] for t in outputs],nodes=nodes)
before=model(old,op,oc);after=model(g,pa,nc);write(N/'frontier-original.blif',before);write(N/'frontier-new.blif',after)
mutant=copy.deepcopy(after);q=mutant['nodes'][after['outputs'][0]];q['mask']^=(1<<(1<<len(q['inputs'])))-1;write(N/'frontier-tampered.blif',mutant)
for label,right,want in [('equivalence','frontier-new.blif','Networks are equivalent.'),('negative','frontier-tampered.blif','NOT EQUIVALENT')]:
 s=N/(label+'.abc');s.write_text('cec frontier-original.blif '+right+'\n',encoding='ascii')
 with (N/(label+'.log')).open('x',encoding='utf8') as log:
  child=subprocess.Popen([sys.executable,str(Path(__file__).with_name('run_installed_abc.py')),'--script',str(s)],cwd=N,stdout=log,stderr=subprocess.STDOUT)
  try:rc=child.wait(timeout=180)
  except subprocess.TimeoutExpired:(N/(label+'-stop.json')).write_text(json.dumps(stop_launched_tree(child)));raise
 body=(N/(label+'.log')).read_text();assert rc==0 and want in body,(label,rc,body)
 if label=='equivalence':assert 'NOT EQUIVALENT' not in body
cut=dict(old_cells=sorted(oc),new_cells=sorted(nc),outputs=outputs,boundary=boundary,copies=copies,tokens={v:k for k,v in names.items()})
(N/'frontier-cut.json').write_text(json.dumps(cut,indent=2));(N/'frontier-proof-graph.pkl').write_bytes(pickle.dumps((key,old,op,g,pa,cut)))
inputs={str(f):sha(f) for f in [Path(__file__),P/'native_source.v',P/'native-graph-audit.json',P/'weight-zero-independent-audit.json',N/'native_source.v',N/'shared-frontier-edit.json',N/'graft.json',O/'plan.json',O/'control-candidates.json',O/'rgb-cut.json',O/'balanced.blif',O/'boundary.pkl',N/'frontier-original.blif',N/'frontier-new.blif',N/'frontier-tampered.blif',N/'frontier-cut.json',N/'equivalence.log',N/'negative.log',V/'current-functional-same-edge-1.json']}
dest=N/'shared-frontier-independent-audit.json';proof=dict(status='SHARED_FRONTIER_COMPOSITIONAL_CEC_PASS_NOT_PHYSICAL',source_sha256=sha(N/'native_source.v'),parent_sha256=key,primitive_count=len(actual),old_cut_primitives=len(oc),new_cut_luts=len(nc),all_boundary_inputs=len(boundary),all_exposed_outputs=len(outputs),all_boolean_inputs=True,tampered_output_rejected=True,exact_local_copies=len(copies),all_original_sequential_state_exact=True,ports_protected_assigns_exact=True,extra_cycles=0,arbitrary_X_equivalence=False,inputs=inputs,seconds=time.monotonic()-start,hardware_access=False,board_release=False)
dest.write_text(json.dumps(proof,indent=2));(N/'zero-source-parse.pkl').write_bytes(pickle.dumps((sha(N/'native_source.v'),actual,pa)))
wp=json.loads((P/'weight-zero-independent-audit.json').read_text());wp.update(source_sha256=sha(N/'native_source.v'),shared_frontier_proof=str(dest),scope='Same-cycle exact complete Boolean cut CEC plus exact local copies; all original sequential/arithmetical state preserved. Actual vendor model and current mapped tests required.',inputs={**wp['inputs'],**inputs,str(dest):sha(dest)});(N/'weight-zero-independent-audit.json').write_text(json.dumps(wp,indent=2))
na=json.loads((P/'native-graph-audit.json').read_text());na.update(retained_primitives=len(actual),shared_frontier_scope=proof['status'],inputs={**na['inputs'],**inputs,str(dest):sha(dest)});(N/'native-graph-audit.json').write_text(json.dumps(na,indent=2));print(json.dumps({k:v for k,v in proof.items() if k!='inputs'}),flush=True)
