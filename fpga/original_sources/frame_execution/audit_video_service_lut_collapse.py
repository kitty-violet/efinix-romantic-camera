"""Independent changed-cell graph/parameter proof and official-model binding."""
import argparse,json,time
from pathlib import Path
from run_units import ROOT,sha
from audit_continuous_vectors import expand_graph
from probe_full_native_split_map import params
from complete_native_logic_split import protected_blocks
from video_service_graph import load

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp')
 start=time.monotonic();graft=json.loads((root/'graft.json').read_text());c=graft['lut_composition'];base=Path(c['base'])
 assert sha(base/'native_source.v')==c['base_sha256'] and sha(c['proof'])==c['proof_sha256'] and sha(c['survey'])==c['survey_sha256']
 pf=json.loads(Path(c['proof']).read_text());sv=json.loads(Path(c['survey']).read_text());assert c['pairs']==[sv['rows'][i] for i in pf['accepted']]
 assert pf['vectors_per_pair']==256 and pf['status']=='OFFICIAL_LUT_FOUR_STATE_FILTER_COMPLETED'
 for f,h in pf['inputs'].items():assert sha(f)==h
 old_audit=json.loads((base/'native-graph-audit.json').read_text())
 for f,h in old_audit['inputs'].items():assert sha(f)==h
 D=load();oldgraft=json.loads((base/'graft.json').read_text());edits={}
 for r in oldgraft['changed_loads']:edits.setdefault((r['cell'],r['pin']),{})[tuple(r['from_token'])]=tuple(r['to_token'])
 removed=set(oldgraft['removed']);old={n:(k,{p:[edits.get((n,p),{}).get(t,t) for t in ts] for p,ts in pins.items()}) for n,(k,pins) in D['g'].items() if n not in removed}
 s=(root/'native_source.v').read_text();new=expand_graph(s);np=params(s);parents={r['parent'] for r in c['pairs']};children={r['child']:r for r in c['pairs']}
 assert set(new)==set(old)-parents and len(new)==graft['retained_primitive_count']
 for n,(kind,pins) in new.items():
  if n in children:
   r=children[n];expected={f'I{i}':[tuple(t)] for i,t in enumerate(r['inputs'])};expected['O']=[tuple(r['output'])]
   assert kind=='EFX_LUT4' and pins==expected and np[n]=={'LUTMASK':(16,r['lutmask'])},n
   assert D['ps'][r['parent']]['LUTMASK'][1]==r['original_parent_mask'] and D['ps'][n]['LUTMASK'][1]==r['original_child_mask']
  else:assert (kind,pins)==old[n] and np.get(n)==D['ps'].get(n),n
 assert protected_blocks(s)==protected_blocks((base/'native_source.v').read_text())
 for f in base.glob('*.sv'):assert sha(f)==sha(root/f.name)
 # Exterior text, including behavioral RTL and all existing assignments,
 # must be byte-identical when public primitive instances/parameters removed.
 import re
 from native_local_drivers import CELL
 def exterior(t):
  t=CELL.sub('',t);t=re.sub(r'^\s*defparam\s+[^;]*;','',t,flags=re.M);return re.sub(r'\s+','',t)
 assert exterior(s)==exterior((base/'native_source.v').read_text())
 result=dict(status='SERVICE_NATIVE_RETAINED_GRAPH_AND_CUT_PASS',retained_primitives=len(new),removed_primitives=len(graft['removed']),
  explicit_changed_loads=len(graft['changed_loads']),equivalent_lut_compositions=len(children),four_state_vectors=256*len(children),
  ordinary_retained_primitives_exact=len(new)-len(children),state_and_latency_unchanged=True,protected_blocks_unchanged=True,existing_assignments_unchanged=True,
  inputs={str(f):sha(f) for f in [Path(__file__),base/'native-graph-audit.json',base/'native_source.v',root/'native_source.v',root/'graft.json',Path(c['proof']),Path(c['survey'])]},
  seconds=time.monotonic()-start,hardware_access=False,board_release=False)
 out=root/'native-graph-audit.json';assert not out.exists();out.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='inputs'},indent=2))
if __name__=='__main__':main()
