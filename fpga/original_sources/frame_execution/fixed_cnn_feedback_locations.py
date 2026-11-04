"""Retain verified queue/mux locations, matching functions rather than LUT IDs."""
import itertools,json,re
from pathlib import Path
from run_units import ROOT,EF,sha
from audit_canonical_multiseed import audit
P='video_processing_pipeline/pipeline/cnn/core/dut/engine/'
def norm(s):return re.sub(r'\s+','',s).replace('\\','')
def cells(path):
 text=path.read_text();allcells={}
 for kind,name,body in re.findall(r'^\s*(EFX_SRL8|EFX_LUT4)\s+(\S+)\s*\((.*?)\) /\*',text,re.M|re.S):
  ports={k:norm(v) for k,v in re.findall(r'\.(\w+)\(([^()]*)\)',body)}
  if name.lstrip('\\').startswith(P+'rq/QUEUE[0].BIT[') or ports.get('O','').startswith(P+'skip/feedback_rhs['):
   allcells[name.lstrip('\\')]=dict(kind=kind,ports=ports,params={})
 for name,key,value in re.findall(r'^\s*defparam\s+(\S+)\s*\.(\w+)\s*=\s*([^;]+);',text,re.M):
  name=name.lstrip('\\')
  if name in allcells:allcells[name]['params'][key]=norm(value)
 return allcells
def signature(c):
 if c['kind']=='EFX_SRL8':return c
 pins=c['ports'];inputs=sorted({v for k,v in pins.items() if k.startswith('I')})
 assert len(inputs)==3 and set(c['params'])=={'LUTMASK'}
 mask=int(c['params']['LUTMASK'].split("'h")[1],16);truth=[]
 # Also prove independence of every unused physical LUT input.
 for bits in itertools.product((0,1),repeat=3):
  values=dict(zip(inputs,bits));indices=[sum(values[pins['I'+str(i)]]<<i for i in range(3)) | (unused<<3) for unused in (0,1)]
  assert (mask>>indices[0]&1)==(mask>>indices[1]&1)
  truth.append(mask>>indices[0]&1)
 return dict(kind=c['kind'],inputs=inputs,truth=truth,output=pins['O'])
def generate(root,reference):
 root=Path(root);reference=Path(reference);prior=audit(reference);assert prior['current_timing_pass']
 old=Path(prior['physical']);oldnet=old/'outflow/ti60f225_oob.map.v';newnet=root/'outflow/ti60f225_oob.map.v'
 before=cells(oldnet);after=cells(newnet)
 oldmux={c['ports']['O']:n for n,c in before.items() if c['kind']=='EFX_LUT4'}
 newmux={c['ports']['O']:n for n,c in after.items() if c['kind']=='EFX_LUT4'}
 expected={P+f'skip/feedback_rhs[{i}]' for i in range(256)}
 assert set(oldmux)==set(newmux)==expected
 mapping={newmux[o]:oldmux[o] for o in expected}
 for i in range(256):
  name=P+f'rq/QUEUE[0].BIT[{i}].storage';assert before[name]['kind']==after[name]['kind']=='EFX_SRL8';mapping[name]=name
 for new,prev in mapping.items():assert signature(after[new])==signature(before[prev]),new
 locations={w[0]:tuple(map(int,w[1:])) for line in (old/'outflow/ti60f225_oob.out.placeloc').read_text().splitlines() if len(w:=line.split())==4 and not line.startswith('#')}
 assigned={n:locations[prev] for n,prev in mapping.items()};assert len(assigned)==512 and len(set(assigned.values()))==512
 path=root/'fixed_dsp.placeloc';base_hash=sha(path);original=path.read_text()
 existing={w[0]:tuple(map(int,w[1:])) for line in original.splitlines() if len(w:=line.split())==4 and not line.startswith('#')}
 assert not set(existing).intersection(assigned)
 assert not {v[:2] for v in existing.values()}.intersection(v[:2] for v in assigned.values())
 path.write_text(original+'# Function-matched authored CNN queue and residual mux; no protected cells.\n'+''.join(f'{n} {x} {y} {s}\n' for n,(x,y,s) in sorted(assigned.items())))
 # The combined placement file is frozen only after both placement generators.
 cp=root/'cnn_ram_placement.json';cm=json.loads(cp.read_text());assert cm['assignments_sha256']==base_hash
 cm['base_assignments_sha256']=base_hash;cm['assignments_sha256']=sha(path);cp.write_text(json.dumps(cm,indent=2))
 result=dict(status='CNN_FEEDBACK_FUNCTION_MATCHED_LOCATIONS_NOT_TIMING_PASS',reference=str(reference),reference_result_sha256=sha(reference/'result.json'),reference_placement_sha256=sha(old/'outflow/ti60f225_oob.out.placeloc'),old_map_sha256=sha(oldnet),new_map_sha256=sha(newnet),base_assignments_sha256=base_hash,assignments_sha256=sha(path),mapping=mapping,assignments=assigned,primitive_count=512,truth_table_cases=256*8,clock_or_logic_changed=False,generator_sha256=sha(__file__),docs={n:sha(EF/'doc/topics'/n) for n in ('loc-primitives.html','loc-rules.html')})
 (root/'cnn_feedback_placement.json').write_text(json.dumps(result,indent=2));(root/'cnn_feedback_generator.py').write_bytes(Path(__file__).read_bytes());return result
def verify_placed(root):
 root=Path(root);r=json.loads((root/'cnn_feedback_placement.json').read_text());reference=Path(r['reference']);prior=audit(reference)
 assert prior['current_timing_pass'] and sha(reference/'result.json')==r['reference_result_sha256']
 old=Path(prior['physical']);assert sha(old/'outflow/ti60f225_oob.out.placeloc')==r['reference_placement_sha256']
 assert sha(root/'outflow/ti60f225_oob.map.v')==r['new_map_sha256'] and sha(old/'outflow/ti60f225_oob.map.v')==r['old_map_sha256']
 assert sha(root/'fixed_dsp.placeloc')==r['assignments_sha256']
 oldcells=cells(old/'outflow/ti60f225_oob.map.v');newcells=cells(root/'outflow/ti60f225_oob.map.v')
 for n,prev in r['mapping'].items():assert signature(newcells[n])==signature(oldcells[prev])
 actual={w[0]:list(map(int,w[1:])) for line in (root/'outflow/ti60f225_oob.out.placeloc').read_text().splitlines() if len(w:=line.split())==4 and not line.startswith('#')}
 for n,pos in r['assignments'].items():assert actual[n]==pos,(n,actual[n],pos)
 return dict(status='CNN_FEEDBACK_512_LOCATIONS_MATCH',placement_sha256=sha(root/'outflow/ti60f225_oob.out.placeloc'),proof_sha256=sha(root/'cnn_feedback_placement.json'),board_release=False)
