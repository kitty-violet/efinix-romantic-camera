"""Check mapped stage elimination and changed public fanout before placement."""
import argparse,json,pickle
from pathlib import Path
from run_units import ROOT,sha
from continuous_control_native import connections
from audit_video_service_rq_map import expand_cached_ports

def load(b):
 mapped=b/'outflow/ti60f225_oob.map.v';key,_,g,_,p=pickle.loads((b/'retained-audit-parse.pkl').read_bytes())
 assert key[1]==sha(mapped)
 expand_cached_ports(g,mapped.read_text());d,l=connections(g)
 return g,p,d,l
def main():
 p=argparse.ArgumentParser()
 for n in ('board','native','output'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();b=a.board.resolve();n=a.native.resolve();out=a.output.resolve();assert out.is_relative_to(ROOT/'tmp') and not out.exists()
 r=json.loads((n/'ram-address-edit.json').read_text());survey=Path(r['survey']);sr=json.loads(survey.read_text());old=Path(sr['board'])
 assert sha(survey)==r['survey_sha256'] and sr['map_sha256']==sha(old/'outflow/ti60f225_oob.map.v')
 full=json.loads((b/'retained-map-audit.json').read_text());assert full['status']=='VIDEO_SERVICE_RETAINED_MAPPED_GRAPH_PASS' and full['conflicts']==0
 g0,p0,d0,l0=load(old);g,p,d,l=load(b);rows=[];tokens=set()
 # Other members of this same45-bit transformation may consume a shared
 # parent on a different LUT input pin. Account for every known edit, not
 # just the one child currently being inspected; keep exact pin identity.
 parent_tokens={g[v['parent']][1]['O'][0] for v in sr['candidates']}
 expected_loads={t:set(l0[t]) for t in parent_tokens}
 for v in sr['candidates']:
  child=v['child']
  for pin,ts in g0[child][1].items():
   if pin=='O':continue
   for t in ts:
    if t in expected_loads:expected_loads[t].discard((child,pin))
  for pin,ts in g[child][1].items():
   if pin=='O':continue
   for t in ts:
    if t in expected_loads:expected_loads[t].add((child,pin))
 for v in sr['candidates']:
  child,up=v['child'],v['parent'];assert g[child][0]==g[up][0]=='EFX_LUT4'
  assert g[up]==g0[up] and p[up]==p0[up]
  parent_out=g[up][1]['O'][0];assert parent_out==g0[up][1]['O'][0]
  oldins={t for pin,ts in g0[child][1].items() if pin!='O' for t in ts}
  newins={t for pin,ts in g[child][1].items() if pin!='O' for t in ts}
  assert parent_out in oldins and parent_out not in newins
  wanted={tuple(t) for t in v['inputs']};assert newins==wanted
  assert g[child][1]['O']==g0[child][1]['O']
  output=g[child][1]['O'][0];assert l[output]==l0[output] and any(g[x][0]=='EFX_RAM10' and pin=='RADDR' for x,pin in l[output])
  assert l[parent_out]==expected_loads[parent_out] and l[parent_out],(child,up,l[parent_out]-expected_loads[parent_out],expected_loads[parent_out]-l[parent_out])
  tokens.update(wanted);rows.append(dict(child=child,parent=up,remaining_parent_loads=len(l[parent_out]),ram_sinks=sorted(l[output]),removed_logic_levels=1))
 fanout=[]
 for token in sorted(tokens,key=str):
  fanout.append(dict(net=token,driver=d.get(token),before=len(l0[token]),after=len(l[token]),delta=len(l[token])-len(l0[token])))
 result=dict(status='ACTUAL_45_RAM_ADDRESS_STAGES_AND_FANOUT_CHECKED_PASS',map_sha256=sha(b/'outflow/ti60f225_oob.map.v'),address_bits=len(rows),
  rows=rows,fanout=fanout,max_changed_net_fanout=max(x['after'] for x in fanout),
  scope='Exact same output sinks and unchanged shared parent gates. One address-output LUT dependency removed at all45 bits; no physical placement/delay claim.',
  evidence={str(f):sha(f) for f in (Path(__file__),n/'ram-address-edit.json',survey,b/'retained-map-audit.json',b/'outflow/ti60f225_oob.map.v',old/'outflow/ti60f225_oob.map.v')},hardware_access=False,board_release=False)
 out.write_text(json.dumps(result,indent=2));print(json.dumps(dict(status=result['status'],address_bits=len(rows),max_fanout=result['max_changed_net_fanout'],changed=[v for v in fanout if v['delta']>0])))
if __name__=='__main__':main()
