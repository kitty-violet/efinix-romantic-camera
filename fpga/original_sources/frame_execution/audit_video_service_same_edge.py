"""Independent delta proof against frozen61, including legal-state APB induction."""
import argparse,copy,json,pickle,re,time
from pathlib import Path
from run_units import ROOT,sha
from native_local_drivers import graph
from probe_full_native_split_map import params
from complete_native_logic_split import protected_blocks
from control_remap_cut import boundaries
p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);a=p.parse_args();N=a.native.resolve();r=json.loads((N/'same-edge-edit.json').read_text());P=Path(r['parent']);start=time.monotonic()
assert r['source_sha256']==sha(N/'native_source.v') and r['parent_sha256']==sha(P/'native_source.v')
for f,h in r['inputs'].items():assert sha(f)==h
key,old,op=pickle.loads((P/'zero-source-parse.pkl').read_bytes());assert key==r['parent_sha256']
text=(N/'native_source.v').read_text();g=graph(text);pa=params(text);parent_text=(P/'native_source.v').read_text()
alias={};copies={}
for row in r['copies']:
 n=row['copy'];base=row['original'];src=tuple(row['old_output']);dst=tuple(row['new_output']);assert dst not in alias;alias[dst]=src;copies[n]=base
 assert g[n][0]==old[base][0]=='EFX_LUT4' and pa[n]==op[base]
 assert g[n][1]['O']==[dst] and old[base][1]['O']==[src]
 def norm(ts):return [alias.get(t,t) for t in ts]
 # A copy of the skip driver can itself use a later copied prefix; checked below after full alias collection.
def root(t):
 seen=set()
 while t in alias:assert t not in seen;seen.add(t);t=alias[t]
 return t
def normpins(pins,output=True):return {p:([root(t) for t in ts] if output or p!='O' else ts) for p,ts in pins.items()}
for n,base in copies.items():assert normpins(g[n][1])==old[base][1],n
for row in r['copies']:
 dst=tuple(row['new_output'])
 actual={(n,p) for n,(k,pins) in g.items() for p,ts in pins.items() if p!='O' and dst in ts}
 expected=set(map(tuple,row['sinks']));assert actual==expected,(row['copy'],actual^expected)
V=ROOT/'tmp/board-test/video-service-review1';s=json.loads((V/'service62-fusion-survey.json').read_text());f=json.loads((ROOT/'tmp/frame-execution/video-service62-fusions1/result.json').read_text());assert f['survey_sha256']==sha(V/'service62-fusion-survey.json')
for path,h in f['inputs'].items():assert sha(path)==h
accepted={s['rows'][i]['target']:s['rows'][i] for i in f['accepted']};assert set(accepted)==set(r['fusions']) and len(accepted)==32
truth=0
for n,row in accepted.items():
 for sn,(kind,pins) in row['old_graph'].items():assert old[sn]==(kind,{p:[tuple(t) for t in ts] for p,ts in pins.items()});assert op[sn]=={k:tuple(v) for k,v in row['old_parameters'][sn].items()}
 kind,pins=row['new_graph'][n];assert g[n][0]==kind and normpins(g[n][1])=={p:[tuple(t) for t in ts] for p,ts in pins.items()}
 assert pa[n]=={k:tuple(v) for k,v in row['new_parameters'][n].items()};truth+=256
removed=set(old)-set(g);dead=set(json.loads((V/'control-priority-dead-cone-cost.json').read_text())['removed']);assert removed==dead|{'LUT__133025','LUT__141150'}
targets,_,_=boundaries(old,parent_text,dead);assert not targets
apb_d=('net','service62_apb_next',None)
assert g['SERVICE62_APB_NEXT']==('EFX_LUT4',{'I0':old['cpu/state_oh[4]~FF'][1]['D'],'I1':old['ceg_net4871~FF'][1]['Q'],'I2':old['ceg_net4871~FF'][1]['D'],'O':[apb_d]})
assert pa['SERVICE62_APB_NEXT']=={'LUTMASK':(16,0x8080)}
assert g['SERVICE62_APB_ENABLE']==('EFX_FF',{'D':[apb_d],'CE':[('literal',"1'b1",None)],'CLK':[('net','process_clk_150',None)],'SR':[('net','w_arstn',None)],'Q':[('net','apb_enable',None)]})
assert pa['SERVICE62_APB_ENABLE']==op['ceg_net4871~FF']
pr_path=ROOT/'tmp/frame-execution/video-service-apb-projection-screen1/result.json';pr=json.loads(pr_path.read_text());assert pr['status']=='CPU_ONEHOT_AND_SAME_EDGE_APB_PROJECTION_PROVED_NOT_IMPLEMENTED' and pr['source_sha256']==key
assert pr['additional_external_cycles']==0 and pr['all_state_next_inputs_unconstrained_except_proved_onehot_invariant']
for path,h in pr['inputs'].items():assert sha(path)==h
assert set(g)-set(old)==set(copies)|{'SERVICE62_APB_NEXT','SERVICE62_APB_ENABLE'}
same=0
for n in old.keys()&g.keys():
 if n in accepted:continue
 assert g[n][0]==old[n][0] and normpins(g[n][1])==old[n][1] and pa[n]==op[n],n;same+=1
assert protected_blocks(text)==protected_blocks(parent_text)
for pat in (r'^\s*(?:input|output|inout)\b.*$',r'^\s*assign\b.*$'):assert re.findall(pat,text,re.M)==re.findall(pat,parent_text,re.M)
for path in P.glob('*.sv'):assert sha(path)==sha(N/path.name)
# Detect accidental consumers of removed private output nets; APB output intentionally retained by new FF.
for n in removed-{'LUT__133025','LUT__141150'}:
 t=old[n][1]['O'][0];assert not any(t in ts for kind,pins in g.values() for pin,ts in pins.items() if pin!='O'),n
paths=[Path(__file__),P/'native_source.v',P/'native-graph-audit.json',P/'weight-zero-independent-audit.json',N/'native_source.v',N/'same-edge-edit.json',N/'graft.json',pr_path,V/'service62-fusion-survey.json',ROOT/'tmp/frame-execution/video-service62-fusions1/result.json',V/'current-functional-control-priority-1.json']
inputs={str(p):sha(p) for p in paths};dest=N/'same-edge-independent-audit.json'
proof=dict(status='SAME_EDGE_ENABLE_COMPOSITIONAL_EQUIVALENCE_PASS_NOT_PHYSICAL',source_sha256=sha(N/'native_source.v'),parent_sha256=key,primitive_count=len(g),removed_primitives=len(removed),new_primitives=len(set(g)-set(old)),unchanged_or_exact_alias_primitives=same,fused_luts=len(accepted),vendor_four_state_vectors=truth,exact_four_state_copies=len(copies),new_projection_ffs=1,legal_state_induction=True,all_original_sequential_state_preserved=True,external_cycles_added=0,ports_protected_assigns_exact=True,independent_full_source_reparse=True,arbitrary_illegal_cpu_states_equivalent=False,inputs=inputs,seconds=time.monotonic()-start,hardware_access=False,board_release=False)
dest.write_text(json.dumps(proof,indent=2));(N/'zero-source-parse.pkl').write_bytes(pickle.dumps((sha(N/'native_source.v'),g,pa)))
wp=json.loads((P/'weight-zero-independent-audit.json').read_text());wp.update(source_sha256=sha(N/'native_source.v'),same_edge_proof=str(dest),scope='Exact compositional equivalence to61 after proved CPU reset/one-hot invariant; all original data/weight state preserved, zero added external cycles.',inputs={**wp['inputs'],**inputs,str(dest):sha(dest)})
(N/'weight-zero-independent-audit.json').write_text(json.dumps(wp,indent=2))
na=json.loads((P/'native-graph-audit.json').read_text());na.update(retained_primitives=len(g),same_edge_scope=proof['status'],inputs={**na['inputs'],**inputs,str(dest):sha(dest)})
(N/'native-graph-audit.json').write_text(json.dumps(na,indent=2));print(json.dumps({k:v for k,v in proof.items() if k!='inputs'}))
