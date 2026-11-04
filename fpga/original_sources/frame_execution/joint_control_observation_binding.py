"""Observation-only legal-state bindings; original mapped design remains untouched."""
import copy,json,pickle
from pathlib import Path
from run_units import sha
from continuous_control_native import connections

def bind_apb(board,native,g,ps):
 evidence={}
 def bind(f):
  r=json.loads(f.read_text());evidence[str(f)]=sha(f)
  for p,h in r.get('inputs',{}).items():assert sha(p)==h,p;evidence[p]=h
  return r
 c=bind(native/'joint-control-independent-audit.json');assert c['status']=='JOINT_CONTROL_COMPOSITIONAL_CEC_PASS_NOT_PHYSICAL' and c['source_sha256']==sha(board/'native_source.v')
 assert c['all_original_sequential_state_exact'] and c['extra_cycles']==0 and c['address_projection_induction']
 parent=Path(json.loads((native/'joint-control-edit.json').read_text())['parent']);f=bind(parent/'shared-frontier-independent-audit.json');assert f['source_sha256']==c['parent_sha256']==sha(parent/'native_source.v')
 previous=Path(json.loads((parent/'shared-frontier-edit.json').read_text())['parent']);p=bind(previous/'same-edge-independent-audit.json');assert p['source_sha256']==f['parent_sha256']==sha(previous/'native_source.v')
 assert p['legal_state_induction'] and p['external_cycles_added']==0
 n='SERVICE62_APB_ENABLE';states=[]
 for root in (previous,parent,native):
  k,a,pa=pickle.loads((root/'zero-source-parse.pkl').read_bytes());assert k==sha(root/'native_source.v');states.append((a[n],pa[n]))
 assert states[0]==states[1]==states[2]
 retained=bind(board/'retained-map-audit.json');assert retained['status']=='VIDEO_SERVICE_RETAINED_MAPPED_GRAPH_PASS' and retained['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v')
 kind,pins=g[n];assert kind=='EFX_FF' and pins['Q']==[('net','apb_enable',None)] and pins['CLK']==[('net','process_clk_150',None)] and ps[n]['SR_VALUE']==(1,0)
 for f in (Path(__file__),native/'joint-control-edit.json',board/'outflow/ti60f225_oob.map.v'):evidence[str(f)]=sha(f)
 return dict(old_cell='LUT__133025',current_cell=n,token=pins['Q'][0],legal_reset_same_edge_equivalence=True,cycles_added=0,evidence=evidence)

def normalize_address_observation(new):
 board=new['path'].parent.parent;native=Path(json.loads((board/'inputs.json').read_text())['native']);edit=json.loads((native/'joint-control-edit.json').read_text());parent=Path(edit['parent']);proof=json.loads((native/'joint-control-independent-audit.json').read_text())
 assert proof['source_sha256']==sha(board/'native_source.v') and proof['address_projection_induction'] and proof['extra_cycles']==0
 for f,h in proof['inputs'].items():assert sha(f)==h
 plan=json.loads((Path(edit['plan_root'])/'plan.json').read_text())['address'];key,old,op=pickle.loads((parent/'zero-source-parse.pkl').read_bytes());nk,actual,ap=pickle.loads((native/'zero-source-parse.pkl').read_bytes());assert key==edit['parent_sha256'] and nk==edit['source_sha256']
 g,ps=new['g'],new['p'];mapping={}
 for n in plan['source_ffs']:
  assert g[n][0]=='EFX_FF' and ps[n]==op[n];mapping[old[n][1]['Q'][0]]=g[n][1]['Q'][0]
 selected={v['old_output_cell']:v for v in plan['projection']}
 for i,n in enumerate(plan['owned']):
  if n in selected:
   ff=selected[n]['new_ff'];assert g[ff][0]=='EFX_FF' and ps[ff]==ap[ff];mapping[old[n][1]['O'][0]]=g[ff][1]['Q'][0]
  else:mapping[old[n][1]['O'][0]]=('net','observation_only_address_'+str(i),None)
 for n in plan['owned']:
  name='OBSERVATION_ONLY_'+n;assert name not in g
  g[name]=('EFX_LUT4',{p:[mapping[t] if t[0]=='net' else t for t in ts] for p,ts in old[n][1].items()});ps[name]=op[n]
 for row in plan['projection']:g.pop(row['new_ff'])
 new['d']=connections(g)[0]
 return dict(status='OBSERVATION_BOUNDARY_NORMALIZED_ONLY_AFTER_SAME_EDGE_PROOF',projected_ff_count=len(selected),extra_cycles=0,evidence={str(f):sha(f) for f in [Path(__file__),native/'joint-control-independent-audit.json',native/'joint-control-edit.json',parent/'native_source.v',native/'native_source.v',board/'retained-map-audit.json']})
