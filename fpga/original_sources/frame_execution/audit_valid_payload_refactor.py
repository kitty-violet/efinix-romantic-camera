"""Bind valid-only arithmetic and optional retained-repeat cache to board sources."""
import argparse,copy,json,re
from pathlib import Path
from run_units import ROOT,sha
from audit_resource_scope import read_board
from audit_control_payload_refactor import gate
from audit_fixed_datapath_resources import verify_fixed_datapath,quant_gate
from audit_direct_sram_response import block_lines

ARITH={'shared_rq_oc8_cluster.sv','shared_pool_banked_fifo.sv','requantize_i32_mul18_q17_pipeline.sv'}

def verify_valid_payload(binding):
 assert binding['status']=='VALID_PAYLOAD_RESOURCE_BINDING_PASS'
 variant=binding['variant'];assert variant in ('quant_safety','retained_repeat')
 changed=ARITH|({'unified_column_reader.sv'} if variant=='retained_repeat' else set())
 before=read_board(Path(binding['reference_board']));after=read_board(Path(binding['board']))
 assert len(after['names'])==125 and before['names'].keys()==after['names'].keys()
 assert {n for n in after['names'] if after['names'][n]!=before['names'][n]}==changed
 assert binding['changed_sources']==sorted(changed)
 for key in ('inputs_sha256','map_sha256'):assert binding[key]==after['result'][key]
 parent=binding['inherited_binding'];assert sha(parent['path'])==parent['sha256']
 prior=json.loads(Path(parent['path']).read_text());verify_fixed_datapath(prior)
 assert Path(prior['board'])==Path(binding['reference_board'])
 roles={'quant','cluster','reference_cluster','guard','pool'}|({'reader'} if variant=='retained_repeat' else set())
 assert {g['role'] for g in binding['gates']}==roles
 roots={}
 for rec in binding['gates']:
  role=rec['role'];root=Path(rec['path']);roots[role]=root
  names,actual=quant_gate(root) if role=='quant' else gate(root,role in ('cluster','reference_cluster'))
  assert rec==dict(actual,role=role)
  expected=before['names'] if role=='reference_cluster' else after['names']
  common=names.keys()&expected.keys()
  required={'requantize_i32_mul18_q17_pipeline.sv'} if role=='quant' else {'unified_column_reader.sv'} if role=='reader' else ARITH if role=='pool' else changed
  assert required<=common and all(names[n]==expected[n] for n in common),(role,common)
  result=json.loads((root/'result.json').read_text());log=(root/'simulation.log').read_text()
  assert not re.search(r'\bFAIL\b|\*\* (Error|Fatal)|%Error',log)
  if role=='pool':
   assert result['status']=='POOL_OVERFLOW_FOUR_STATE_PASS'
   assert result['line']=='PASS POOL_OVERFLOW rejected=16 recovered=16 held=128 banks=4 signs=2 reset_cancel=1'
  elif role=='guard':
   assert result['line']=='PASS DATAPATH_GUARD cases=13 ar=11 aw=1 r=10 w=1 b=1'
   assert result['active_lines']==['PASS DATAPATH_ACTIVE_CANCEL simultaneous_compute_header_dma=1 restored_words=3 restored_acks=3']
   meta=json.loads((root/'inputs.json').read_text());assert sha(root/'obj/Vtb_cnn_datapath_guard.exe')==meta['binary_sha256']
  elif role=='cluster':
   assert re.search(r'PASS COMPUTE_CLUSTER .*blocks=24 .*scalar_checks=70336 .*mac_held=[1-9]\d* ',log)
   assert 'PASS RESIDUAL_ABORT_PROBE stages=5 direction_fault=1' in log
  elif role=='reader':
   assert result['status']=='RETAINED_REPEAT_FOUR_STATE_PACKET_PASS'
   assert 'cases=120 layouts=28 rejected_layouts=3' in result['line'] and 'cancels=9' in result['line']
   assert result['reference_sha256']==before['names']['unified_column_reader.sv']
   assert result['candidate_sha256']==after['names']['unified_column_reader.sv']
 old,new=block_lines(roots['reference_cluster']),block_lines(roots['cluster'])
 assert len(old)==len(new)==24 and binding['block_cycles']==new
 # A resource candidate may not buy space by making any covered block slower.
 for a,b in zip(old,new):
  assert re.sub(r'cycles=\d+|last_command=\d+','n',a)==re.sub(r'cycles=\d+|last_command=\d+','n',b)
  for field in ('cycles','last_command'):
   assert int(re.search(r'\b'+field+r'=(\d+)',b)[1])<=int(re.search(r'\b'+field+r'=(\d+)',a)[1]),(a,b)
 if variant=='quant_safety':assert old==new
 vectors=[]
 for role in ('cluster','reference_cluster'):
  root=roots[role];meta=json.loads((root/'frozen/inputs.json').read_text());vectors.append(meta['vectors'])
  for name,digest in meta['vectors'].items():assert sha(root/'frozen/vectors'/name)==digest
 assert vectors[0]==vectors[1]
 # Packet rejection precedes both SRAM queue write enables, unchanged from reference.
 text=(Path(binding['board'])/'assembly/shared_pool_banked_fifo.sv').read_text()
 original=(Path(binding['reference_board'])/'assembly/shared_pool_banked_fifo.sv').read_text()
 assert text==original.replace('.BANK_PREFIX_HISTORY(1)) rq(','.BANK_PREFIX_HISTORY(1),.REJECT_BIAS_OVERFLOW(1)) rq(')
 assert all(binding[k] is False for k in ('resource_fit','physical_timing_pass','board_download_allowed')) and binding['frame_ms'] is None
 return before,after

def main():
 p=argparse.ArgumentParser();p.add_argument('--variant',choices=('quant_safety','retained_repeat'),required=True)
 for name in ('board','reference-board','quant','cluster','reference-cluster','guard','pool','reader'):
  p.add_argument('--'+name,type=Path,required=name!='reader')
 a=p.parse_args();board=a.board.resolve();ref=a.reference_board.resolve()
 assert board.is_relative_to(ROOT/'tmp') and not (board/'gate_binding.json').exists()
 after=read_board(board);roles={'quant','cluster','reference_cluster','guard','pool'}|({'reader'} if a.variant=='retained_repeat' else set());records=[]
 for role in sorted(roles):
  root=getattr(a,role).resolve();_,rec=quant_gate(root) if role=='quant' else gate(root,role in ('cluster','reference_cluster'));records.append(dict(rec,role=role))
 binding=dict(status='VALID_PAYLOAD_RESOURCE_BINDING_PASS',variant=a.variant,board=str(board),reference_board=str(ref),
  changed_sources=sorted(ARITH|({'unified_column_reader.sv'} if a.variant=='retained_repeat' else set())),source_names=125,
  inputs_sha256=after['result']['inputs_sha256'],map_sha256=after['result']['map_sha256'],
  inherited_binding=dict(path=str(ref/'gate_binding.json'),sha256=sha(ref/'gate_binding.json')),gates=records,
  block_cycles=block_lines(a.cluster),resource_fit=False,physical_timing_pass=False,board_download_allowed=False,frame_ms=None)
 before,after=verify_valid_payload(binding);checks=[]
 for label in ('map_identity','changed_sources','required_gate','cycle_evidence','release_flag'):
  bad=copy.deepcopy(binding)
  if label=='map_identity':bad['map_sha256']='0'*64
  elif label=='changed_sources':bad['changed_sources']=[]
  elif label=='required_gate':bad['gates']=bad['gates'][:-1]
  elif label=='cycle_evidence':bad['block_cycles']=bad['block_cycles'][:-1]
  else:bad['resource_fit']=True
  try:verify_valid_payload(bad)
  except AssertionError:checks.append(label)
  else:raise AssertionError('Tampering accepted '+label)
 binding['resource_delta']={k:v-before['result']['resources'][k] for k,v in after['result']['resources'].items()}
 (board/'gate_binding.json').write_text(json.dumps(binding,indent=2)+'\n')
 (board/'binding_checks.json').write_text(json.dumps(dict(rejected_tampering=checks),indent=2))
 print(json.dumps(dict(status=binding['status'],variant=a.variant,delta=binding['resource_delta'],tamper_checks=checks)))
if __name__=='__main__':main()
