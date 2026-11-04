"""Bind real parameter RAM, numeric schedule and active recovery to board bytes."""
import argparse,json,re
from pathlib import Path
from run_units import ROOT,sha
from audit_resource_scope import read_board
from audit_control_payload_refactor import gate
from audit_fixed_storage import verify_fixed_storage
CHANGED={'next_header_prefetch.sv','shared_pool_banked_fifo.sv','compute_column_cluster.sv','cnn_frame_datapath.sv'}
ROLES={'cache','header','cluster','reference_cluster','guard','guard_fourstate'}

def verify_header_ram(b):
 assert b['status']=='HEADER_RAM_RESOURCE_BINDING_PASS'
 before=read_board(Path(b['reference_board']));after=read_board(Path(b['board']))
 assert len(after['names'])==125 and before['names'].keys()==after['names'].keys()
 assert {n for n in after['names'] if after['names'][n]!=before['names'][n]}==CHANGED
 assert b['changed_sources']==sorted(CHANGED)
 for k in ('inputs_sha256','map_sha256'):assert b[k]==after['result'][k]
 parent=b['inherited_binding'];assert sha(parent['path'])==parent['sha256']
 prior=json.loads(Path(parent['path']).read_text());verify_fixed_storage(prior)
 assert Path(prior['board'])==Path(b['reference_board'])
 assert {g['role'] for g in b['gates']}==ROLES
 roots={}
 for rec in b['gates']:
  role=rec['role'];root=Path(rec['path']);roots[role]=root
  names,actual=gate(root,native=role in ('cluster','reference_cluster'))
  assert rec==dict(actual,role=role)
  expected=before['names'] if role=='reference_cluster' else after['names']
  common=names.keys()&expected.keys()
  for n in common:assert names[n]==expected[n],(role,n)
  required=CHANGED if role.startswith('guard') else {'next_header_prefetch.sv'} if role=='header' else {'compute_column_cluster.sv','shared_pool_banked_fifo.sv'} if role in ('cluster','reference_cluster') else set()
  assert required<=common
  result=json.loads((root/'result.json').read_text());log=(root/'simulation.log').read_text()
  assert not re.search(r'\bFAIL\b|\*\* (Error|Fatal)|%Error',log)
  if role=='cache':
   assert result['status']=='HEADER_GROUP_CACHE_REAL_RAM_FOUR_STATE_PASS'
   assert result['line']=='PASS HEADER_GROUP_CACHE checks=56952 firsts=7168 first_stalls=0 groups=1,2,4 cancel=1 active_write_reject=1 concurrent_next_bank=1'
   cache=(root/'header_group_cache3.sv').read_text()
   rows=json.loads((Path(b['board'])/'inputs.json').read_text())['inputs']
   embedded=Path(next(r['snapshot'] for r in rows if Path(r['snapshot']).name=='compute_column_cluster.sv')).read_text()
   assert embedded.endswith(cache),'Tested cache must exactly match embedded production RAM'
  elif role=='header':
   assert result['status']=='STREAM_HEADER_SHARED_REAL_RESIDENT_RAM_PASS'
   assert result['line']=='PASS HEADER_PREFETCH headers=12 bytes=2432 weight_words=1500 faults=6 held=84 cycles=18813'
   assert 'PASS HEADER_GROUP_RAM_SCAN checks=336' in log
   assert 'PASS COMPACT_SCALE_FIELDS cases=4 positive_negative_valid_invalid=1' in log
  elif role in ('cluster','reference_cluster'):
   assert result['status']=='NATIVE_LOCAL_CLUSTER_NUMERIC_PROTOCOL_PASS'
   assert result['line']=='PASS COMPUTE_CLUSTER seed=71 blocks=24 scalar_checks=70336 held=7281 mac_held=1330 faults=2 switches=5'
  else:
   assert result['line']=='PASS DATAPATH_GUARD cases=13 ar=11 aw=1 r=10 w=1 b=1'
   assert result['active_lines']==['PASS DATAPATH_ACTIVE_CANCEL simultaneous_compute_header_dma=1 restored_words=3 restored_acks=3']
   if role=='guard':
    meta=json.loads((root/'inputs.json').read_text())
    assert sha(root/'obj/Vtb_cnn_datapath_guard.exe')==meta['binary_sha256']
 # The isolated cache and end-to-end header test must use identical RAM RTL.
 assert sha(roots['cache']/'header_group_cache3.sv')==sha(roots['header']/'header_group_cache3.sv')
 def blocks(root):return re.findall(r'^BLOCK .*$',(root/'simulation.log').read_text(),re.M)
 old,new=blocks(roots['reference_cluster']),blocks(roots['cluster'])
 assert len(old)==len(new)==24 and old==new
 assert b['equal_block_cycle_rows']==24
 assert all(b[k] is False for k in ('resource_fit','physical_timing_pass','board_download_allowed')) and b['frame_ms'] is None
 return before,after

def main():
 p=argparse.ArgumentParser()
 for n in ('board','reference-board',*[r.replace('_','-') for r in sorted(ROLES)]):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();board=a.board.resolve();ref=a.reference_board.resolve()
 assert board.is_relative_to(ROOT/'tmp') and not (board/'gate_binding.json').exists()
 after=read_board(board);records=[]
 for role in sorted(ROLES):
  _,rec=gate(getattr(a,role).resolve(),native=role in ('cluster','reference_cluster'));records.append(dict(rec,role=role))
 b=dict(status='HEADER_RAM_RESOURCE_BINDING_PASS',board=str(board),reference_board=str(ref),changed_sources=sorted(CHANGED),source_names=125,
  inputs_sha256=after['result']['inputs_sha256'],map_sha256=after['result']['map_sha256'],
  inherited_binding=dict(path=str(ref/'gate_binding.json'),sha256=sha(ref/'gate_binding.json')),gates=records,equal_block_cycle_rows=24,
  resource_fit=False,physical_timing_pass=False,board_download_allowed=False,frame_ms=None)
 before,after=verify_header_ram(b)
 b['resource_delta']={k:v-before['result']['resources'][k] for k,v in after['result']['resources'].items()}
 (board/'gate_binding.json').write_text(json.dumps(b,indent=2)+'\n');print(json.dumps(b['resource_delta']))
if __name__=='__main__':main()
