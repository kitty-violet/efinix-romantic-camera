"""Compose current tap projection with the frozen64 observation proofs."""
import json,pickle
from pathlib import Path
from run_units import ROOT,sha
from continuous_control_native import connections
import joint_control_observation_binding as parent_binding

def current(board,native):
 p=json.loads((native/'tap-project-independent-audit.json').read_text());e=json.loads((native/'tap-project-edit.json').read_text());parent=Path(e['parent'])
 assert p['status']=='TAP_PROJECT_COMPLETE_GRAPH_INDUCTION_PASS_NOT_PHYSICAL' and p['source_sha256']==sha(board/'native_source.v')==sha(native/'native_source.v')
 assert p['all_other_graph_and_parameters_exact'] and p['projection_induction'] and p['extra_cycles']==0 and p['projected_ffs']==5
 evidence={str(f):sha(f) for f in [Path(__file__),native/'tap-project-independent-audit.json',native/'tap-project-edit.json',board/'retained-map-audit.json',board/'outflow/ti60f225_oob.map.v']}
 for f,h in p['inputs'].items():assert sha(f)==h;evidence[f]=h
 m=json.loads((board/'retained-map-audit.json').read_text());assert m['status']=='VIDEO_SERVICE_RETAINED_MAPPED_GRAPH_PASS' and m['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v')
 return p,e,parent,evidence

def bind_apb(board,native,g,ps):
 p,e,parent,evidence=current(board,native)
 oldboard=ROOT/'tmp/frame-execution/connected-board-video-service-probe55-joint-control'
 b=parent_binding.bind_apb(oldboard,parent,g,ps);b['evidence'].update(evidence);return b

def normalize_tap_observation(new):
 board=new['path'].parent.parent;native=Path(json.loads((board/'inputs.json').read_text())['native']);p,e,parent,evidence=current(board,native)
 ad=json.loads((Path(e['plan_root'])/'plan.json').read_text())['address'];key,old,op=pickle.loads((parent/'zero-source-parse.pkl').read_bytes());nk,actual,ap=pickle.loads((native/'zero-source-parse.pkl').read_bytes());assert key==p['parent_sha256'] and nk==p['source_sha256']
 g,ps=new['g'],new['p'];mapping={}
 for n in ad['source_ffs']:assert g[n][0]=='EFX_FF' and ps[n]==op[n];mapping[old[n][1]['Q'][0]]=g[n][1]['Q'][0]
 selected={v['old_output_cell']:v for v in ad['projection']}
 for i,n in enumerate(ad['owned']):
  if n in selected:
   f=selected[n]['new_ff'];assert g[f][0]=='EFX_FF' and ps[f]==ap[f];mapping[old[n][1]['O'][0]]=g[f][1]['Q'][0]
  else:mapping[old[n][1]['O'][0]]=('net','observation_only_tap_'+str(i),None)
 for n in ad['owned']:
  name='OBSERVATION_ONLY_TAP_'+n;assert name not in g;g[name]=('EFX_LUT4',{q:[mapping[t] if t[0]=='net' else t for t in ts] for q,ts in old[n][1].items()});ps[name]=op[n]
 for row in ad['projection']:g.pop(row['new_ff'])
 new['d']=connections(g)[0]
 return dict(status='OBSERVATION_BOUNDARY_NORMALIZED_ONLY_AFTER_SAME_EDGE_PROOF',projected_ff_count=5,extra_cycles=0,evidence=evidence)

def normalize_address_observation(new):
 board=new['path'].parent.parent;native=Path(json.loads((board/'inputs.json').read_text())['native']);p,e,parent,evidence=current(board,native)
 oldpath=new['path'];new['path']=ROOT/'tmp/frame-execution/connected-board-video-service-probe55-joint-control/outflow/ti60f225_oob.map.v'
 try:b=parent_binding.normalize_address_observation(new)
 finally:new['path']=oldpath
 b['evidence'].update(evidence);return b
