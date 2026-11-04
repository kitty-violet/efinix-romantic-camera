"""Bind dispatch routing/storage trials to cycle tests and whole-board map."""
import argparse,copy,json,re
from pathlib import Path
from run_units import ROOT,sha
from audit_resource_scope import read_board
from audit_control_payload_refactor import gate
from audit_direct_sram_response import verify_direct_binding,block_lines

CHANGED={'unified_column_dispatch32.sv'}
ROLES={'unit','cluster','reference_cluster','fourstate','guard','guard_fourstate'}

def verify_radix3_binding(binding):
 assert binding['status'] in ('RADIX3_DISPATCH_SOURCE_CYCLE_BINDING_PASS','DISPATCH_STORAGE_SOURCE_CYCLE_BINDING_PASS')
 before=read_board(Path(binding['reference_board']));after=read_board(Path(binding['board']))
 assert before['names'].keys()==after['names'].keys() and len(after['names'])==125
 assert {n for n in after['names'] if after['names'][n]!=before['names'][n]}==CHANGED
 assert binding['changed_sources']==sorted(CHANGED)
 variant=binding.get('variant','radix3')
 assert variant in ('radix3','shared_history','shared_existing_taps')
 source=(Path(binding['board'])/'assembly/unified_column_dispatch32.sv').read_text()
 if variant=='shared_existing_taps':assert 'localparam ALIAS_A=' in source
 elif variant=='shared_history':assert 'PW/RGB accept an input' in source and 'localparam ALIAS_A=' not in source
 else:assert 'begin:ROTATE3' in source
 assert not after['result']['new_undriven_signatures']
 for key in ('inputs_sha256','map_sha256'):assert binding[key]==after['result'][key]
 inherited=binding['inherited_binding'];assert sha(inherited['path'])==inherited['sha256']
 parent=json.loads(Path(inherited['path']).read_text());verify_direct_binding(parent)
 assert Path(parent['board'])==Path(binding['reference_board'])
 assert {r['role'] for r in binding['gates']}==ROLES
 roots={}
 for rec in binding['gates']:
  role=rec['role'];root=Path(rec['path']);roots[role]=root
  names,actual=gate(root,role in ('cluster','reference_cluster'));assert rec==dict(actual,role=role)
  expected=before['names'] if role=='reference_cluster' else after['names']
  common=names.keys() & expected.keys();assert CHANGED<=common
  assert all(names[n]==expected[n] for n in common),(role,common)
  result=json.loads((root/'result.json').read_text());log=(root/'simulation.log').read_text()
  if role=='unit':
   assert result['status']=='UNIFIED_DISPATCH_CYCLE_EQUIVALENCE_PASS'
   assert re.search(r'cases=40 owner_tokens=43348 held=8446',log)
  elif role in ('cluster','fourstate'):
   assert 'PASS RESIDUAL_ABORT_PROBE stages=5 direction_fault=1' in log
   if role=='cluster':assert re.search(r'PASS COMPUTE_CLUSTER .*blocks=24 .*scalar_checks=70336 .*mac_held=[1-9]\d* ',log)
   else:assert re.search(r'RESIDUAL_RETURN_HOLD checks=[1-9]\d*',log)
  elif role in ('guard','guard_fourstate'):
   assert result['active_lines'] and 'PASS DATAPATH_ACTIVE_CANCEL' in log
 old,new=block_lines(roots['reference_cluster']),block_lines(roots['cluster'])
 assert old==new and len(new)==24 and binding['block_cycles']==new
 vectors=[]
 for role in ('reference_cluster','cluster'):
  root=roots[role];meta=json.loads((root/'frozen/inputs.json').read_text());vectors.append(meta['vectors'])
  for name,digest in meta['vectors'].items():assert sha(root/'frozen/vectors'/name)==digest
 assert vectors[0]==vectors[1]
 assert all(binding[k] is False for k in ('resource_fit','physical_timing_pass','board_download_allowed'))
 assert binding['frame_ms'] is None
 return before,after

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--variant',choices=('radix3','shared_history','shared_existing_taps'),default='radix3')
 for name in ('board','reference-board',*[r.replace('_','-') for r in sorted(ROLES)]):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();board=a.board.resolve();ref=a.reference_board.resolve()
 assert board.is_relative_to(ROOT/'tmp') and not (board/'gate_binding.json').exists()
 after=read_board(board);records=[]
 for role in sorted(ROLES):
  _,r=gate(getattr(a,role).resolve(),role in ('cluster','reference_cluster'));records.append(dict(r,role=role))
 binding=dict(status='DISPATCH_STORAGE_SOURCE_CYCLE_BINDING_PASS',variant=a.variant,board=str(board),reference_board=str(ref),
  changed_sources=sorted(CHANGED),source_names=125,inputs_sha256=after['result']['inputs_sha256'],
  map_sha256=after['result']['map_sha256'],inherited_binding=dict(path=str(ref/'gate_binding.json'),sha256=sha(ref/'gate_binding.json')),
  gates=records,block_cycles=block_lines(a.cluster),resource_fit=False,physical_timing_pass=False,
  frame_ms=None,board_download_allowed=False)
 before,after=verify_radix3_binding(binding);checks=[]
 for label in ('map_identity','changed_sources','required_gate','cycle_evidence'):
  wrong=copy.deepcopy(binding)
  if label=='map_identity':wrong['map_sha256']='0'*64
  elif label=='changed_sources':wrong['changed_sources']=[]
  elif label=='required_gate':wrong['gates']=wrong['gates'][:-1]
  else:wrong['block_cycles']=wrong['block_cycles'][:-1]
  try:verify_radix3_binding(wrong)
  except AssertionError:checks.append(label)
  else:raise AssertionError('Binding accepted tampering: '+label)
 binding['resource_delta']={k:v-before['result']['resources'].get(k,0) for k,v in after['result']['resources'].items()}
 (board/'binding_checks.json').write_text(json.dumps(dict(rejected_tampering=checks),indent=2))
 (board/'gate_binding.json').write_text(json.dumps(binding,indent=2)+'\n')
 print(json.dumps(dict(status=binding['status'],delta=binding['resource_delta'],identical_blocks=24,tamper_checks=checks)))

if __name__=='__main__':main()
