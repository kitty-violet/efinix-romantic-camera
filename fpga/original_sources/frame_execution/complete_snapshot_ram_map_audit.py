"""Complete the RAM-only step after the preserved argv-wrapper failure.

Existing actual-map truth/finite-probe checks are reused only with exact hashes.
No RTL, mapping output, old log or failed-run status is overwritten.
"""
import argparse,json,shutil
from pathlib import Path
from run_units import ROOT,HERE,sha
from snapshot_ram_board import audit_delta
from snapshot_ram_cdc import structure

def main():
 p=argparse.ArgumentParser()
 for name in ('board','root','source'):p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();board=a.board.resolve();root=a.root.resolve();source=a.source.resolve()
 assert root.is_relative_to(ROOT/'tmp/frame-execution') and not root.exists()
 audit_delta(board);mapped=sha(board/'outflow/ti60f225_oob.map.v')
 r=json.loads((source/'result.json').read_text())
 assert r['status']=='FAULT_PROBE_ACTUAL_CONTROL_BOOLEAN_AND_CDC_MAP_PASS'
 assert r['map_sha256']==mapped and r['board_inputs_sha256']==sha(board/'inputs.json')
 assert r['checks']==17504 and r['whole_control_upstream_shared_cells']==0
 for path,digest in r['evidence'].items():assert sha(path)==digest,path
 expected={'new_diagnostic_cells.json':'SPLIT_OUTPUTS_FULL_MAP_ZERO_COMB4_PASS',
           'held_data_structure.json':'OFP5_HELD_DATA_GATE_AND_RESET_CHAINS_MAPPED_PASS'}
 for name,status in expected.items():
  data=json.loads((source/name).read_text())
  assert data['status']==status,(name,data['status'])
  assert data['map_sha256']==mapped
  checker='audit_split_logic_outputs_map.py' if name=='new_diagnostic_cells.json' else 'split_logic_outputs_cdc.py'
  assert data['auditor_sha256']==sha(HERE/checker)
 b=json.loads((source/'boolean_result.json').read_text())
 assert b['checks']==17504 and b['mismatch_count']==0 and b['map_sha256']==mapped
 s=structure(board)
 root.mkdir()
 for f in source.iterdir():
  assert f.is_file();shutil.copy2(f,root/f.name)
  assert sha(f)==sha(root/f.name)
 (root/'snapshot_ram_structure.json').write_text(json.dumps(s,indent=2))
 (root/'ram_wrapper.json').write_text(json.dumps(dict(status='RAM_MAP_AUDIT_COMPLETED_FROM_VERIFIED_PREFIX',
  source=str(source),source_files={str(f):sha(f) for f in source.iterdir()},
  wrapper_sha256=sha(__file__),map_sha256=mapped,
  preserved_failure=str(source.parent/'map-audit.log'),hardware_access=False),indent=2))
 print('RAM_MAP_AUDIT_COMPLETED_FROM_VERIFIED_PREFIX',flush=True)
if __name__=='__main__':main()
