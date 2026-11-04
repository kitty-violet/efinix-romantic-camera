"""Independent exact retained-graph and explicit cut audit; no hardware."""
import argparse,json,re,time
from pathlib import Path
from video_service_graph import load,SOURCE
from run_units import ROOT,sha
from audit_continuous_vectors import expand_graph
from probe_full_native_split_map import params
from complete_native_logic_split import protected_blocks
from camera_joint_native import OUT

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
 start=time.monotonic();proof=json.loads((root/'graft.json').read_text());D=load();old=D['g'];s=(root/'native_source.v').read_text()
 new=expand_graph(s);np=params(s);removed=set(proof['removed']);assert set(new)==set(old)-removed
 edits={}
 for r in proof['changed_loads']:edits.setdefault((r['cell'],r['pin']),{})[tuple(r['from_token'])]=tuple(r['to_token'])
 for n,(k,pins) in new.items():
  expect={p:[edits.get((n,p),{}).get(t,t) for t in ts] for p,ts in old[n][1].items()}
  assert k==old[n][0] and pins==expect,('unexpected retained primitive change',n)
  assert np.get(n)==D['ps'].get(n),('retained parameter change',n)
 exported={tuple(r['token']) for rows in proof['exports'].values() for r in rows}|{tuple(r['token']) for r in proof['supplied']}
 missing=[];crossings=0
 for n in removed:
  for p,ts in old[n][1].items():
   if p not in OUT[old[n][0]]:continue
   for t in ts:
    for cell,pin in D['loads'][t]:
     if cell not in removed and edits.get((cell,pin),{}).get(t,t)==t:
      crossings+=1
      if t not in exported:missing.append((n,t,cell,pin))
 assert not missing,missing[:20]
 assert protected_blocks(s)==protected_blocks(SOURCE.read_text())
 # Preserve preexisting assignments; the service only appends new bindings.
 assigns=lambda text:re.findall(r'^\s*assign\b[^;]+;',text,re.M)
 old_assigns=assigns(SOURCE.read_text());new_assigns=assigns(s)
 assert all(x in new_assigns for x in old_assigns)
 result=dict(status='SERVICE_NATIVE_RETAINED_GRAPH_AND_CUT_PASS',retained_primitives=len(new),
  removed_primitives=len(removed),explicit_changed_loads=len(proof['changed_loads']),boundary_loads=crossings,
  protected_blocks_unchanged=True,existing_assignments_unchanged=True,hardware_access=False,board_release=False,
  seconds=time.monotonic()-start,inputs={str(p):sha(p) for p in [SOURCE,root/'native_source.v',root/'graft.json',Path(__file__)]})
 f=root/'native-graph-audit.json';assert not f.exists();f.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='inputs'},indent=2))
if __name__=='__main__':main()
