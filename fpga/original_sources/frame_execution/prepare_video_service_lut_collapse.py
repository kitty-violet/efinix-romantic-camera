"""Apply only officially four-state-equivalent LUT pair compositions."""
import argparse,collections,json,re,shutil,time
from pathlib import Path
from run_units import ROOT,sha
from native_local_drivers import CELL,render_token

def main():
 p=argparse.ArgumentParser();p.add_argument('--proof',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 start=time.monotonic();proof_path=a.proof.resolve();out=a.root.resolve();assert out.is_relative_to(ROOT/'tmp') and not out.exists()
 proof=json.loads(proof_path.read_text());assert proof['status']=='OFFICIAL_LUT_FOUR_STATE_FILTER_COMPLETED'
 for f,h in proof['inputs'].items():assert sha(f)==h,f
 survey_path=Path(proof['survey']);assert sha(survey_path)==proof['survey_sha256'];survey=json.loads(survey_path.read_text())
 base=Path(survey['native']);assert sha(base/'native_source.v')==proof['source_sha256']==survey['native_sha256']
 parent_audit=base/'native-graph-audit.json';old_audit=json.loads(parent_audit.read_text())
 assert old_audit['status']=='SERVICE_NATIVE_RETAINED_GRAPH_AND_CUT_PASS'
 for f,h in old_audit['inputs'].items():assert sha(f)==h,f
 rows=[survey['rows'][i] for i in proof['accepted']];assert rows and not set(proof['accepted'])&set(proof['rejected'])
 remove={r['parent'] for r in rows};change={r['child']:r for r in rows};assert len(remove)==len(change)==len(rows) and not remove&change.keys()
 text=(base/'native_source.v').read_text();seen=collections.Counter();records=[]
 def replace(m):
  name=m[2].lstrip('\\')
  if name not in remove and name not in change:return m[0]
  assert m[1]=='EFX_LUT4';seen[name]+=1
  if name in remove:new=''
  else:
   r=change[name];pins=[f'.I{i}({render_token(tuple(t))})' for i,t in enumerate(r['inputs'])]
   pins.append('.O('+render_token(tuple(r['output']))+')')
   new='\n    EFX_LUT4 \\'+name+' ('+', '.join(pins)+') /* exact four-state LUT composition */ ;'
  records.append(dict(name=name,before=m[0],after=new));return new
 composed=CELL.sub(replace,text);assert set(seen)==remove|change.keys() and all(v==1 for v in seen.values())
 def parameter(m):
  name=m[1].lstrip('\\')
  if name in remove:return ''
  if name in change:
   assert m[2]=='LUTMASK';return f"\n    defparam \\{name} .LUTMASK = 16'h{change[name]['lutmask']:04x};"
  return m[0]
 composed=re.sub(r'^\s*defparam\s+(\S+?)\s*\.([A-Z_0-9]+)\s*=[^;]*;',parameter,composed,flags=re.M)
 out.mkdir()
 for f in base.glob('*.sv'):shutil.copy2(f,out/f.name)
 (out/'native_source.v').write_text(composed,encoding='utf8')
 graft=json.loads((base/'graft.json').read_text());assert not remove&set(graft['removed'])
 graft['removed']=sorted(set(graft['removed'])|remove);graft['removed_kinds']['EFX_LUT4']+=len(remove)
 graft['changed_loads']=[r for r in graft['changed_loads'] if r['cell'] not in remove]
 graft['retained_primitive_count']-=len(remove)
 graft['lut_composition']=dict(base=str(base),base_sha256=sha(base/'native_source.v'),proof=str(proof_path),proof_sha256=sha(proof_path),
  survey=str(survey_path),survey_sha256=sha(survey_path),pairs=rows,changed_state_elements=0,latency_cycles_added=0,accepted_four_state=True)
 for f in (Path(__file__),proof_path,survey_path,parent_audit):graft['inputs'][str(f)]=sha(f)
 (out/'graft.json').write_text(json.dumps(graft,indent=2))
 (out/'composition-edit-records.json').write_text(json.dumps(records,indent=2))
 print(json.dumps(dict(status='FOUR_STATE_LUT_COMPOSITION_APPLIED_PENDING_INDEPENDENT_GRAPH_AUDIT',pairs=len(rows),source_sha256=sha(out/'native_source.v'),seconds=time.monotonic()-start,hardware_access=False)))

if __name__=='__main__':main()
