"""Bind ring-word DMA and compact header storage to the same complete board."""
import argparse,json,re
from pathlib import Path
from run_units import ROOT,sha
from audit_resource_scope import read_board
from audit_control_payload_refactor import gate
from audit_valid_payload_refactor import verify_valid_payload
CHANGED={'frame_arena_row_dma.sv','next_header_prefetch.sv'}
ROLES={'dma','header','guard','guard_fourstate'}

def verify_fixed_storage(b):
 assert b['status']=='FIXED_STORAGE_RESOURCE_BINDING_PASS'
 before=read_board(Path(b['reference_board']));after=read_board(Path(b['board']))
 assert len(after['names'])==125 and before['names'].keys()==after['names'].keys()
 assert {n for n in after['names'] if after['names'][n]!=before['names'][n]}==CHANGED
 assert b['changed_sources']==sorted(CHANGED)
 for k in ('inputs_sha256','map_sha256'):assert b[k]==after['result'][k]
 parent=b['inherited_binding'];assert sha(parent['path'])==parent['sha256']
 prior=json.loads(Path(parent['path']).read_text());verify_valid_payload(prior)
 assert Path(prior['board'])==Path(b['reference_board'])
 assert {g['role'] for g in b['gates']}==ROLES
 for rec in b['gates']:
  role=rec['role'];root=Path(rec['path']);names,actual=gate(root);assert rec==dict(actual,role=role)
  common=names.keys()&after['names'].keys()
  required=CHANGED if role.startswith('guard') else {'frame_arena_row_dma.sv'} if role=='dma' else {'next_header_prefetch.sv'}
  assert required<=common
  meta=json.loads((root/'inputs.json').read_text());result=json.loads((root/'result.json').read_text())
  for row in meta['inputs']:
   n=Path(row['snapshot']).name
   if n not in common:continue
   if role=='dma':
    assert row['source_sha256']==after['names'][n] and sha(row['source'])==after['names'][n]
    assert Path(row['snapshot']).read_text()==Path(row['source']).read_text()
   else:assert names[n]==after['names'][n],(role,n)
  if role=='dma':
   assert result['status']=='DIRECT_SRAM_DMA_FOUR_STATE_CYCLE_EQUIVALENCE_PASS'
   assert result['outputs']==25 and result['dual_real_ram'] and len(result['runs'])==3
   assert result['candidate_sha256']==after['names']['frame_arena_row_dma.sv']
   assert result['reference_sha256']==before['names']['frame_arena_row_dma.sv']
   for run in result['runs']:
    log=root/f"simulation_{run['seed']}.log";assert sha(log)==run['log_sha256'] and run['edges']>16000
   for n in ('frame_arena_row_dma','tensor_arena12'):
    old=(Path(b['reference_board'])/'assembly'/f'{n}.sv').read_text().replace(n,n+'_reference')
    if n=='tensor_arena12':old=old.replace('tensor_dual_bank','tensor_dual_bank_reference')
    assert (root/f'{n}_reference.sv').read_text()==old
  else:
   log=(root/'simulation.log').read_text();assert not re.search(r'\bFAIL\b|\*\* (Error|Fatal)|%Error',log)
   if role=='header':
    assert result['status']=='COMPACT_HEADER_PREFETCH_SHARED_REAL_RESIDENT_PASS'
    assert 'PASS COMPACT_SCALE_FIELDS cases=4 positive_negative_valid_invalid=1' in log
    assert result['line']=='PASS HEADER_PREFETCH headers=12 bytes=2432 weight_words=1500 faults=6 held=84 cycles=18170'
   else:
    assert result['line']=='PASS DATAPATH_GUARD cases=13 ar=11 aw=1 r=10 w=1 b=1'
    assert result['active_lines']==['PASS DATAPATH_ACTIVE_CANCEL simultaneous_compute_header_dma=1 restored_words=3 restored_acks=3']
    if role=='guard':assert sha(root/'obj/Vtb_cnn_datapath_guard.exe')==meta['binary_sha256']
 assert all(b[k] is False for k in ('resource_fit','physical_timing_pass','board_download_allowed')) and b['frame_ms'] is None
 return before,after

def main():
 p=argparse.ArgumentParser()
 for n in ('board','reference-board',*[r.replace('_','-') for r in sorted(ROLES)]):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();board=a.board.resolve();ref=a.reference_board.resolve()
 assert board.is_relative_to(ROOT/'tmp') and not (board/'gate_binding.json').exists()
 after=read_board(board);records=[]
 for role in sorted(ROLES):
  _,rec=gate(getattr(a,role).resolve());records.append(dict(rec,role=role))
 b=dict(status='FIXED_STORAGE_RESOURCE_BINDING_PASS',board=str(board),reference_board=str(ref),changed_sources=sorted(CHANGED),source_names=125,
  inputs_sha256=after['result']['inputs_sha256'],map_sha256=after['result']['map_sha256'],
  inherited_binding=dict(path=str(ref/'gate_binding.json'),sha256=sha(ref/'gate_binding.json')),gates=records,
  resource_fit=False,physical_timing_pass=False,board_download_allowed=False,frame_ms=None)
 before,after=verify_fixed_storage(b)
 b['resource_delta']={k:v-before['result']['resources'][k] for k,v in after['result']['resources'].items()}
 (board/'gate_binding.json').write_text(json.dumps(b,indent=2)+'\n');print(json.dumps(b['resource_delta']))
if __name__=='__main__':main()
