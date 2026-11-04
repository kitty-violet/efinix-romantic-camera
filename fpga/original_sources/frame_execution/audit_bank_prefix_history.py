"""Bind bank-local prefix storage, its exact simulation projection and board map."""
import argparse,copy,json,re
from pathlib import Path
from run_units import ROOT,HERE,sha
from audit_resource_scope import read_board
from audit_control_payload_refactor import gate
from audit_radix3_dispatch import verify_radix3_binding
from audit_direct_sram_response import block_lines
from prefix_history_projection import project,require_proof

CHANGED={'int8_dual_mac_lane_prefix_stream.sv','shared_mac_stream_front.sv',
 'shared_rq_oc8_cluster.sv','shared_pool_banked_fifo.sv','compute_column_cluster.sv'}
ROLES={'unit','cluster','reference_cluster','fourstate','guard_fourstate'}

def verify_projection_text(original,projected):
 assert projected==project(original),'Simulation projection differs from proved expansion'

def projected_gate(root,native=False):
 names,record=gate(root,native)
 if native:return names,record
 meta=json.loads((root/'inputs.json').read_text())
 if meta.get('prefix_projection'):
  assert meta['prefix_projection']==require_proof()
  rows=[r for r in meta['inputs'] if Path(r['snapshot']).name=='shared_rq_oc8_cluster.sv']
  assert len(rows)==1
  r=rows[0];source=Path(r.get('source',r.get('path')))
  assert sha(source)==r['source_sha256']
  verify_projection_text(source.read_text(),Path(r['snapshot']).read_text())
  names[source.name]=r['source_sha256']
 return names,record

def verify_bank_prefix_binding(binding):
 assert binding['status']=='BANK_PREFIX_SOURCE_CYCLE_BINDING_PASS'
 before=read_board(Path(binding['reference_board']));after=read_board(Path(binding['board']))
 assert len(after['names'])==125 and before['names'].keys()==after['names'].keys()
 assert {n for n in after['names'] if before['names'][n]!=after['names'][n]}==CHANGED
 assert binding['changed_sources']==sorted(CHANGED)
 assert not after['result']['new_undriven_signatures']
 for key in ('inputs_sha256','map_sha256'):assert binding[key]==after['result'][key]
 inherited=binding['inherited_binding'];assert sha(inherited['path'])==inherited['sha256']
 parent=json.loads(Path(inherited['path']).read_text());verify_radix3_binding(parent)
 assert Path(parent['board'])==Path(binding['reference_board'])
 assert binding['projection_proof']==require_proof()
 schedule=binding['schedule_proof'];assert sha(schedule['path'])==schedule['sha256']
 schedule=json.loads(Path(schedule['path']).read_text())
 assert schedule['status']=='BANK_PREFIX_SCHEDULE_DISTANCE_PASS'
 assert schedule['configurations']==4000 and schedule['max_return_distance']==8
 assert schedule['mac_returns']==1219920 and schedule['reuses']==1188160
 for r in schedule['sources']:assert sha(r['path'])==r['sha256']
 assert {r['role'] for r in binding['gates']}==ROLES
 roots={}
 for rec in binding['gates']:
  role=rec['role'];root=Path(rec['path']);roots[role]=root
  names,actual=projected_gate(root,role in ('cluster','reference_cluster'))
  assert rec==dict(actual,role=role)
  expected=before['names'] if role=='reference_cluster' else after['names']
  common=names.keys() & expected.keys()
  assert ({'shared_rq_oc8_cluster.sv'} if role=='unit' else CHANGED)<=common
  assert all(names[n]==expected[n] for n in common),(role,common)
  result=json.loads((root/'result.json').read_text());log=(root/'simulation.log').read_text()
  if role=='unit':
   assert result['status']=='BANK_PREFIX_HISTORY_FOUR_STATE_EQUIVALENCE_PASS'
   assert 'results=7432' in log and 'bypass=1064' in log and 'eviction_rejected=1' in log
  elif role in ('cluster','fourstate'):
   assert 'PASS RESIDUAL_ABORT_PROBE stages=5 direction_fault=1' in log
   if role=='cluster':assert re.search(r'PASS COMPUTE_CLUSTER .*blocks=24 .*scalar_checks=70336 .*mac_held=[1-9]\d* ',log)
   else:assert re.search(r'RESIDUAL_RETURN_HOLD checks=[1-9]\d*',log)
  elif role=='guard_fourstate':
   assert result['line']=='PASS DATAPATH_GUARD cases=13 ar=11 aw=1 r=10 w=1 b=1'
   assert result['active_lines']==['PASS DATAPATH_ACTIVE_CANCEL simultaneous_compute_header_dma=1 restored_words=3 restored_acks=3']
 old,new=block_lines(roots['reference_cluster']),block_lines(roots['cluster'])
 assert len(new)==24 and old==new and binding['block_cycles']==new
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
 for n in ('board','reference-board',*[r.replace('_','-') for r in sorted(ROLES)]):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();board=a.board.resolve();ref=a.reference_board.resolve()
 assert board.is_relative_to(ROOT/'tmp') and not (board/'gate_binding.json').exists()
 after=read_board(board);records=[]
 for role in sorted(ROLES):
  _,r=projected_gate(getattr(a,role).resolve(),role in ('cluster','reference_cluster'))
  records.append(dict(r,role=role))
 schedule=ROOT/'tmp/frame-execution/bank-prefix-schedule1/result.json'
 binding=dict(status='BANK_PREFIX_SOURCE_CYCLE_BINDING_PASS',board=str(board),reference_board=str(ref),
  changed_sources=sorted(CHANGED),source_names=125,inputs_sha256=after['result']['inputs_sha256'],
  map_sha256=after['result']['map_sha256'],inherited_binding=dict(path=str(ref/'gate_binding.json'),sha256=sha(ref/'gate_binding.json')),
  gates=records,projection_proof=require_proof(),schedule_proof=dict(path=str(schedule),sha256=sha(schedule)),
  block_cycles=block_lines(a.cluster),resource_fit=False,physical_timing_pass=False,frame_ms=None,board_download_allowed=False)
 before,after=verify_bank_prefix_binding(binding);checks=[]
 for label in ('map_identity','changed_sources','required_gate','cycle_evidence','projection_proof','schedule_proof'):
  wrong=copy.deepcopy(binding)
  if label=='map_identity':wrong['map_sha256']='0'*64
  elif label=='changed_sources':wrong['changed_sources']=[]
  elif label=='required_gate':wrong['gates']=wrong['gates'][:-1]
  elif label=='cycle_evidence':wrong['block_cycles']=wrong['block_cycles'][:-1]
  else:wrong[label]['sha256']='0'*64
  try:verify_bank_prefix_binding(wrong)
  except AssertionError:checks.append(label)
  else:raise AssertionError('Tampering accepted: '+label)
 # A projected source with a single altered operation is also rejected.
 rq=(board/'assembly/shared_rq_oc8_cluster.sv').read_text()
 projected=project(rq);assert projected.count('assign prior=~(~history_words[selected_age[2:0]]);')==1
 try:verify_projection_text(rq,projected.replace('assign prior=~(~history_words[selected_age[2:0]]);','assign prior=0;'))
 except AssertionError:checks.append('projected_source_exact_regeneration')
 else:raise AssertionError('Altered projection accepted')
 binding['resource_delta']={k:v-before['result']['resources'].get(k,0) for k,v in after['result']['resources'].items()}
 (board/'binding_checks.json').write_text(json.dumps(dict(rejected_tampering=checks),indent=2))
 (board/'gate_binding.json').write_text(json.dumps(binding,indent=2)+'\n')
 print(json.dumps(dict(status=binding['status'],delta=binding['resource_delta'],identical_blocks=24,tamper_checks=checks)))

if __name__=='__main__':main()

