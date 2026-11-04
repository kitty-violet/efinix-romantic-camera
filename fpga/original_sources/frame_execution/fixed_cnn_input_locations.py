"""Retain the complete, function-identical MAC input delay chains."""
import json,re
from pathlib import Path
from run_units import ROOT,sha
from fixed_cnn_feedback_locations import norm
from audit_canonical_multiseed import audit
PREFIX='video_processing_pipeline/pipeline/cnn/core/dut/engine/mac/arithmetic/INPUT_ROW['
def cells(path):
 text=path.read_text();out={}
 for kind,name,body in re.findall(r'^\s*(EFX_SRL8)\s+(\S+)\s*\((.*?)\) /\*',text,re.M|re.S):
  name=name.lstrip('\\')
  if name.startswith(PREFIX):out[name]=dict(kind=kind,ports={k:norm(v) for k,v in re.findall(r'\.(\w+)\(([^()]*)\)',body)},params={})
 for name,key,value in re.findall(r'^\s*defparam\s+(\S+)\s*\.(\w+)\s*=\s*([^;]+);',text,re.M):
  name=name.lstrip('\\')
  if name in out:out[name]['params'][key]=norm(value)
 return out
def match(before,after):
 assert len(before)==len(after)==768 and set(before)==set(after)
 for name,c in before.items():assert c==after[name],name
 return sorted(before)
def locations(path):
 return {w[0]:tuple(map(int,w[1:])) for line in path.read_text().splitlines() if len(w:=line.split())==4 and not line.startswith('#')}
def generate(root,reference):
 root=Path(root);reference=Path(reference);a=audit(reference);assert a['current_timing_pass'];old=Path(a['physical'])
 oldnet=old/'outflow/ti60f225_oob.map.v';newnet=root/'outflow/ti60f225_oob.map.v'
 names=match(cells(oldnet),cells(newnet));op=locations(old/'outflow/ti60f225_oob.out.placeloc')
 assigned={n:op[n] for n in names};assert len(set(assigned.values()))==768
 path=root/'fixed_dsp.placeloc';prior=sha(path);present=locations(path)
 assert not set(present).intersection(assigned)
 assert not set(present.values()).intersection(assigned.values())
 # Whole authored chains are retained, including every EARLY and continuation SRL.
 path.write_text(path.read_text()+'# Function-identical complete MAC input delay chains.\n'+''.join(f'{n} {x} {y} {slot}\n' for n,(x,y,slot) in assigned.items()))
 for name in ('cnn_ram_placement.json','cnn_feedback_placement.json'):
  f=root/name;record=json.loads(f.read_text());assert record['assignments_sha256']==prior
  record['pre_input_assignments_sha256']=prior;record['assignments_sha256']=sha(path);f.write_text(json.dumps(record,indent=2))
 result=dict(status='CNN_INPUT_768_IDENTICAL_LOCATIONS_NOT_TIMING_PASS',reference=str(reference),reference_result_sha256=sha(reference/'result.json'),reference_placement_sha256=sha(old/'outflow/ti60f225_oob.out.placeloc'),old_map_sha256=sha(oldnet),new_map_sha256=sha(newnet),base_assignments_sha256=prior,assignments_sha256=sha(path),assignments=assigned,primitive_count=768,clock_or_logic_changed=False,generator_sha256=sha(__file__))
 (root/'cnn_input_placement.json').write_text(json.dumps(result,indent=2));(root/'cnn_input_generator.py').write_bytes(Path(__file__).read_bytes());return result
def verify_placed(root):
 root=Path(root);r=json.loads((root/'cnn_input_placement.json').read_text());reference=Path(r['reference']);a=audit(reference);assert a['current_timing_pass']
 old=Path(a['physical']);assert sha(reference/'result.json')==r['reference_result_sha256'] and sha(old/'outflow/ti60f225_oob.out.placeloc')==r['reference_placement_sha256']
 assert sha(old/'outflow/ti60f225_oob.map.v')==r['old_map_sha256'] and sha(root/'outflow/ti60f225_oob.map.v')==r['new_map_sha256']
 match(cells(old/'outflow/ti60f225_oob.map.v'),cells(root/'outflow/ti60f225_oob.map.v'))
 assert sha(root/'fixed_dsp.placeloc')==r['assignments_sha256'];actual=locations(root/'outflow/ti60f225_oob.out.placeloc')
 for n,pos in r['assignments'].items():assert actual[n]==tuple(pos),(n,actual[n],pos)
 return dict(status='CNN_INPUT_768_LOCATIONS_MATCH',placement_sha256=sha(root/'outflow/ti60f225_oob.out.placeloc'),proof_sha256=sha(root/'cnn_input_placement.json'),board_release=False)
