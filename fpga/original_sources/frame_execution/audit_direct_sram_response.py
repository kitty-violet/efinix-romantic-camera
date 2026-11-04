"""Bind fixed per-port SRAM contract, cycle evidence and complete board resources."""
import argparse,copy,json,re
from pathlib import Path
from run_units import ROOT,sha
from audit_resource_scope import read_board
from audit_control_payload_refactor import gate
from audit_compact_result_path import verify_compact_binding

CHANGED={'tensor_arena12.sv','unified_column_reader.sv','column_write8.sv','pw_skip_fuse4.sv','frame_arena_row_dma.sv'}
ROLES={'cluster','reference_cluster','fourstate','guard','guard_fourstate','dma','ports'}
CONTRACT='single_outstanding_per_physical_port_retained_router_owner_local_dma_slot3_no_tag_echo'

def block_lines(root):
 return re.findall(r'^BLOCK .*$',(root/'simulation.log').read_text(),re.M)

def verify_direct_binding(binding):
 assert binding['status']=='DIRECT_SRAM_SOURCE_CYCLE_BINDING_PASS'
 assert binding['response_contract']==CONTRACT and binding['internal_tag_fault_detection'] is False
 before=read_board(Path(binding['reference_board']));after=read_board(Path(binding['board']))
 assert before['names'].keys()==after['names'].keys() and len(after['names'])==125
 changed={n for n in after['names'] if after['names'][n]!=before['names'][n]}
 assert changed==CHANGED and binding['changed_sources']==sorted(CHANGED)
 assert not after['result']['new_undriven_signatures']
 for key in ('inputs_sha256','map_sha256'):assert binding[key]==after['result'][key]
 inherited=binding['inherited_binding'];assert sha(inherited['path'])==inherited['sha256']
 parent=json.loads(Path(inherited['path']).read_text());verify_compact_binding(parent)
 assert Path(parent['board'])==Path(binding['reference_board'])
 assert {r['role'] for r in binding['gates']}==ROLES
 roots={}
 for rec in binding['gates']:
  role=rec['role'];root=Path(rec['path']);roots[role]=root
  names,actual=gate(root,role in ('cluster','reference_cluster'));assert rec==dict(actual,role=role)
  expected=before['names'] if role=='reference_cluster' else after['names']
  common=names.keys() & expected.keys();assert common
  if role in ('dma','ports'):
   meta=json.loads((root/'inputs.json').read_text())
   for row in meta['inputs']:
    name=Path(row['snapshot']).name
    if name in common:
     # Unit harnesses use write_text on Windows: only newline normalization
     # is allowed, with the original exact bytes bound to the measured board.
     assert row['source_sha256']==expected[name] and sha(row['source'])==expected[name]
     assert Path(row['snapshot']).read_text()==Path(row['source']).read_text()
  else:assert all(names[n]==expected[n] for n in common),role
  if role in ('guard','guard_fourstate'):assert CHANGED<=common
  if role in ('cluster','fourstate'):assert CHANGED-{'frame_arena_row_dma.sv'}<=common
  result=json.loads((root/'result.json').read_text())
  if role=='dma':
   assert result['status']=='DIRECT_SRAM_DMA_FOUR_STATE_CYCLE_EQUIVALENCE_PASS'
   assert result['dual_real_ram'] and result['internal_tags_removed'] and len(result['runs'])==3
   for run in result['runs']:assert sha(root/f"simulation_{run['seed']}.log")==run['log_sha256']
   for name in ('frame_arena_row_dma','tensor_arena12'):
    old=(Path(binding['reference_board'])/'assembly'/f'{name}.sv').read_text()
    old=old.replace(name,name+'_reference')
    if name=='tensor_arena12':old=old.replace('tensor_dual_bank','tensor_dual_bank_reference')
    assert (root/f'{name}_reference.sv').read_text()==old
  if role=='ports':
   assert result['status']=='DIRECT_SRAM_PORTS_FOUR_STATE_PASS' and len(result['runs'])==6
   for run in result['runs']:assert sha(root/f"{run['top']}_{run['seed']}.log")==run['log_sha256']
  if role in ('cluster','fourstate'):
   log=(root/'simulation.log').read_text()
   assert 'PASS RESIDUAL_ABORT_PROBE stages=5 direction_fault=1' in log
   if role=='cluster':assert re.search(r'PASS COMPUTE_CLUSTER .*blocks=24 .*scalar_checks=70336 .*mac_held=[1-9]\d* ',log)
   else:assert re.search(r'RESIDUAL_RETURN_HOLD checks=[1-9]\d*',log)
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
 for name in ('board','reference-board',*[r.replace('_','-') for r in sorted(ROLES)]):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();board=a.board.resolve();ref=a.reference_board.resolve()
 assert board.is_relative_to(ROOT/'tmp') and not (board/'gate_binding.json').exists()
 after=read_board(board);records=[]
 for role in sorted(ROLES):
  _,r=gate(getattr(a,role).resolve(),role in ('cluster','reference_cluster'));records.append(dict(r,role=role))
 binding=dict(status='DIRECT_SRAM_SOURCE_CYCLE_BINDING_PASS',response_contract=CONTRACT,internal_tag_fault_detection=False,
  board=str(board),reference_board=str(ref),changed_sources=sorted(CHANGED),source_names=125,
  inputs_sha256=after['result']['inputs_sha256'],map_sha256=after['result']['map_sha256'],
  inherited_binding=dict(path=str(ref/'gate_binding.json'),sha256=sha(ref/'gate_binding.json')),
  gates=records,block_cycles=block_lines(a.cluster),resource_fit=False,physical_timing_pass=False,
  frame_ms=None,board_download_allowed=False)
 before,after=verify_direct_binding(binding)
 checks=[]
 for label in ('map_identity','changed_sources','required_gate','contract','cycle_evidence'):
  wrong=copy.deepcopy(binding)
  if label=='map_identity':wrong['map_sha256']='0'*64
  elif label=='changed_sources':wrong['changed_sources']=wrong['changed_sources'][:-1]
  elif label=='required_gate':wrong['gates']=wrong['gates'][:-1]
  elif label=='contract':wrong['internal_tag_fault_detection']=True
  else:wrong['block_cycles']=wrong['block_cycles'][:-1]
  try:verify_direct_binding(wrong)
  except AssertionError:checks.append(label)
  else:raise AssertionError('Binding accepted tampering: '+label)
 binding['resource_delta']={k:v-before['result']['resources'].get(k,0) for k,v in after['result']['resources'].items()}
 (board/'binding_checks.json').write_text(json.dumps(dict(rejected_tampering=checks),indent=2))
 (board/'gate_binding.json').write_text(json.dumps(binding,indent=2)+'\n')
 print(json.dumps(dict(status=binding['status'],delta=binding['resource_delta'],identical_blocks=24,tamper_checks=checks)))

if __name__=='__main__':main()
