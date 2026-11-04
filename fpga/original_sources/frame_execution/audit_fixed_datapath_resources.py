"""Bind fixed-parcel DMA and optional DSP add fusion to their measured board."""
import argparse,copy,json,re
from pathlib import Path
from run_units import ROOT,HERE,sha
from audit_resource_scope import read_board
from audit_control_payload_refactor import gate
from audit_bank_prefix_history import verify_bank_prefix_binding,projected_gate
from audit_direct_sram_response import block_lines
DMA={'frame_arena_row_dma.sv'}
QUANT={'requantize_i32_mul18_q17_pipeline.sv','shared_mac_stream_front.sv'}

def quant_gate(root):
 result=json.loads((root/'result.json').read_text())
 assert result['status']=='COMPACT_QUANT_FOUR_STATE_EQUIVALENCE_PASS'
 assert result['line']=='PASS NARROW_ROUND direct=2297152 pipeline=30830 reset_edges=2'
 assert result['same_four_edge_latency'] and sha(root/'inputs.json')==result['inputs_sha256']
 assert sha(root/'simulation.log')==result['log_sha256']
 meta=json.loads((root/'inputs.json').read_text())
 for name,digest in meta.items():assert sha(name)==digest
 src=next(Path(n) for n in meta if Path(n).name=='requantize_i32_mul18_q17_pipeline.sv' and 'variants' in Path(n).parts)
 assert sha(root/src.name)==meta[str(src)]
 model=next(Path(n) for n in meta if Path(n).name=='efx_dsp48.v')
 assert sha(root/model.name)==meta[str(model)]
 return {src.name:sha(src)},dict(path=str(root),status=result['status'],result_sha256=sha(root/'result.json'),inputs_sha256=sha(root/'inputs.json'))

def verify_fixed_datapath(binding):
 assert binding['status']=='FIXED_DATAPATH_RESOURCE_BINDING_PASS'
 variant=binding['variant'];assert variant in ('fixed_parcel','parcel_quant')
 changed=DMA | (QUANT if variant=='parcel_quant' else set())
 before=read_board(Path(binding['reference_board']));after=read_board(Path(binding['board']))
 assert len(after['names'])==125 and before['names'].keys()==after['names'].keys()
 assert {n for n in after['names'] if before['names'][n]!=after['names'][n]}==changed
 assert binding['changed_sources']==sorted(changed)
 assert not after['result']['new_undriven_signatures']
 for key in ('inputs_sha256','map_sha256'):assert binding[key]==after['result'][key]
 parent=binding['inherited_binding'];assert sha(parent['path'])==parent['sha256']
 parent=json.loads(Path(parent['path']).read_text());verify_bank_prefix_binding(parent)
 assert Path(parent['board'])==Path(binding['reference_board'])
 roles={'dma','dma_guard_fourstate'} | ({'quant','cluster','reference_cluster','guard'} if variant=='parcel_quant' else set())
 assert {r['role'] for r in binding['gates']}==roles
 roots={}
 for rec in binding['gates']:
  role=rec['role'];root=Path(rec['path']);roots[role]=root
  names,actual=quant_gate(root) if role=='quant' else projected_gate(root,role in ('cluster','reference_cluster'))
  assert rec==dict(actual,role=role)
  expected=before['names'] if role=='reference_cluster' else after['names']
  if role=='dma_guard_fourstate':
   expected=dict(before['names']);expected['frame_arena_row_dma.sv']=after['names']['frame_arena_row_dma.sv']
  common=names.keys() & expected.keys()
  if role=='dma':
   # The dual-instance DMA harness uses write_text for module renaming and
   # Windows normalizes copied LF sources to CRLF. Verify exact decoded text;
   # the original snapshot byte hashes remain independently checked by gate.
   for name in common:
    if names[name]!=expected[name]:
     assert (root/name).read_text()==(Path(binding['board'])/'assembly'/name).read_text()
     names[name]=expected[name]
  required=DMA if role in ('dma','dma_guard_fourstate') else {'requantize_i32_mul18_q17_pipeline.sv'} if role=='quant' else QUANT if role in ('cluster','reference_cluster') else changed
  assert required<=common and all(names[n]==expected[n] for n in common),(role,common)
  result=json.loads((root/'result.json').read_text())
  if role=='dma':
   assert result['dual_real_ram'] and result['outputs']==25
   assert result['reference_sha256']==before['names']['frame_arena_row_dma.sv']
   assert result['candidate_sha256']==after['names']['frame_arena_row_dma.sv']
   assert [r['seed'] for r in result['runs']]==[71,391,1027]
   assert sum(r['edges'] for r in result['runs'])==51600
   for r in result['runs']:
    path=root/('simulation_'+str(r['seed'])+'.log');assert sha(path)==r['log_sha256']
    log=path.read_text();assert 'PASS ROW_DMA_SHARED_ABORT transfers=30 failures=21 byte_checks=63910' in log
    assert all('CANCEL_BOUNDARY scenario='+str(i) in log for i in range(7))
    assert not re.search(r'\bFAIL\b|\*\* (Error|Fatal)',log)
  elif role in ('guard','dma_guard_fourstate'):
   assert result['line']=='PASS DATAPATH_GUARD cases=13 ar=11 aw=1 r=10 w=1 b=1'
   assert result['active_lines']==['PASS DATAPATH_ACTIVE_CANCEL simultaneous_compute_header_dma=1 restored_words=3 restored_acks=3']
   if role=='guard':
    meta=json.loads((root/'inputs.json').read_text())
    assert sha(root/'obj/Vtb_cnn_datapath_guard.exe')==meta['binary_sha256']
  elif role=='cluster':
   log=(root/'simulation.log').read_text()
   assert 'blocks=24 scalar_checks=70336 held=7281 mac_held=1330 faults=2 switches=5' in log
   assert 'PASS RESIDUAL_ABORT_PROBE stages=5 direction_fault=1' in log
 if variant=='parcel_quant':
  old,new=block_lines(roots['reference_cluster']),block_lines(roots['cluster'])
  assert len(new)==24 and old==new and new==binding['block_cycles']
  vec=[]
  for role in ('cluster','reference_cluster'):
   root=roots[role];meta=json.loads((root/'frozen/inputs.json').read_text())
   vec.append(meta['vectors'])
   for name,digest in meta['vectors'].items():assert sha(root/'frozen/vectors'/name)==digest
  assert vec[0]==vec[1]
  # The front change only spells out eight constant sign extensions.
  old=(Path(binding['reference_board'])/'assembly/shared_mac_stream_front.sv').read_text()
  new=(Path(binding['board'])/'assembly/shared_mac_stream_front.sv').read_text()
  loop=" for(genvar l=0;l<8;l=l+1)begin:DOT\n  assign dot_data[l*32+:32]={{11{accum[l*32+20]}},accum[l*32+:21]};\n end"
  parts=['{{11{accum['+str(l*32+20)+']}},accum['+str(l*32)+'+:21]}' for l in range(7,-1,-1)]
  expanded=' // Same eight fixed sign extensions without the nested DOT generate alias.\n assign dot_data={'+',\n  '.join(parts)+'};'
  assert old.count(loop)==1 and new==old.replace(loop,expanded)
 assert all(binding[k] is False for k in ('resource_fit','physical_timing_pass','board_download_allowed'))
 assert binding['frame_ms'] is None
 return before,after

def main():
 p=argparse.ArgumentParser();p.add_argument('--variant',choices=('fixed_parcel','parcel_quant'),required=True)
 for n in ('board','reference-board','dma','dma-guard-fourstate','quant','cluster','reference-cluster','guard'):
  p.add_argument('--'+n,type=Path,required=n in ('board','reference-board','dma','dma-guard-fourstate'))
 a=p.parse_args();board=a.board.resolve();ref=a.reference_board.resolve()
 assert board.is_relative_to(ROOT/'tmp') and not (board/'gate_binding.json').exists()
 after=read_board(board);records=[]
 roles={'dma','dma_guard_fourstate'} | ({'quant','cluster','reference_cluster','guard'} if a.variant=='parcel_quant' else set())
 for role in sorted(roles):
  root=getattr(a,role).resolve()
  _,r=quant_gate(root) if role=='quant' else projected_gate(root,role in ('cluster','reference_cluster'))
  records.append(dict(r,role=role))
 binding=dict(status='FIXED_DATAPATH_RESOURCE_BINDING_PASS',variant=a.variant,board=str(board),reference_board=str(ref),
  changed_sources=sorted(DMA | (QUANT if a.variant=='parcel_quant' else set())),source_names=125,
  inputs_sha256=after['result']['inputs_sha256'],map_sha256=after['result']['map_sha256'],
  inherited_binding=dict(path=str(ref/'gate_binding.json'),sha256=sha(ref/'gate_binding.json')),gates=records,
  block_cycles=block_lines(a.cluster) if a.variant=='parcel_quant' else None,
  proof_scope='Four-state DMA (exact decoded source text despite LF/CRLF transport) and quantizer; inherited bank-history and SRAM four-state interfaces; native24-block/active-guard when quantized. No whole-system four-state or physical approval.',
  resource_fit=False,physical_timing_pass=False,board_download_allowed=False,frame_ms=None)
 before,after=verify_fixed_datapath(binding);checks=[]
 for label in ('map_identity','changed_sources','required_gate','release_flag'):
  wrong=copy.deepcopy(binding)
  if label=='map_identity':wrong['map_sha256']='0'*64
  elif label=='changed_sources':wrong['changed_sources']=[]
  elif label=='required_gate':wrong['gates']=wrong['gates'][:-1]
  else:wrong['resource_fit']=True
  try:verify_fixed_datapath(wrong)
  except AssertionError:checks.append(label)
  else:raise AssertionError('Tampering accepted: '+label)
 binding['resource_delta']={k:v-before['result']['resources'].get(k,0) for k,v in after['result']['resources'].items()}
 (board/'binding_checks.json').write_text(json.dumps(dict(rejected_tampering=checks),indent=2))
 (board/'gate_binding.json').write_text(json.dumps(binding,indent=2)+'\n')
 print(json.dumps(dict(status=binding['status'],variant=a.variant,delta=binding['resource_delta'],tamper_checks=checks)))
if __name__=='__main__':main()

