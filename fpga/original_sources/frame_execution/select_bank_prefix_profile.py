"""Select the verified bank-prefix build without changing model or release gates."""
import json
from pathlib import Path
from run_units import ROOT,HERE,sha
from audit_bank_prefix_history import verify_bank_prefix_binding
def rec(p):return dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p))
def main():
 board=ROOT/'tmp/frame-execution/connected-board-bank-prefix1'
 binding=json.loads((board/'gate_binding.json').read_text())
 before,after=verify_bank_prefix_binding(binding)
 old=before['result']['resources'];new=after['result']['resources']
 assert new['LUT4']<old['LUT4']
 assert all(new[k]==old[k] for k in ('RAM10','DPRAM10','DSP24','DSP48'))
 data=json.loads((HERE/'board_resource_candidate.json').read_text())
 assert ROOT/data['selected_board']==Path(binding['reference_board'])
 data['selected_board']=board.relative_to(ROOT).as_posix()
 data['selection_reason']='Four shared bank-local prefix histories replace private snapshots and their wide return mux. Whole 125-source main-chain map: 66175 LUT4 (-3279), 34022 FF (-5276), 2490 SRL8 (+672), unchanged 250 RAM /160 physical DSP. All24 real-RAM block cycles and70336 numeric values unchanged; official SRL unit, proved four-state SRL vector projection, residual abort/hold and active DMA/compute cancel-recovery pass. Capacity, final STA and end-to-end30fps remain unproved.'
 for key,name in [('selected_inputs','inputs.json'),('selected_result','result.json'),('source_binding','gate_binding.json')]:data[key]=rec(board/name)
 overrides={Path(r['path']).name:r for r in data['overrides']}
 for r in json.loads((board/'inputs.json').read_text())['inputs']:
  if Path(r['snapshot']).name in binding['changed_sources']:overrides[Path(r['snapshot']).name]=rec(Path(r['source']))
 data['overrides']=list(overrides.values());seen={r['path'] for r in data['gates']}
 for g in binding['gates']:
  root=Path(g['path']);r=rec(root/'result.json')
  if r['path'] in seen:continue
  result=json.loads((root/'result.json').read_text());r['status']=result['status']
  if (root/'native_inputs.json').is_file():r['input_manifest']=(root/'native_inputs.json').relative_to(ROOT).as_posix()
  data['gates'].append(r);seen.add(r['path'])
 print(json.dumps(data,indent=2))
if __name__=='__main__':main()

