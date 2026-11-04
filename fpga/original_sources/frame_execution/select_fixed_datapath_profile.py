"""Generate a profile proposal after the fixed-datapath source and gate audit."""
import argparse,json
from pathlib import Path
from run_units import ROOT,HERE,sha
from audit_fixed_datapath_resources import verify_fixed_datapath
def rec(p):return dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p))
def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);a=p.parse_args()
 board=a.board.resolve();binding=json.loads((board/'gate_binding.json').read_text())
 before,after=verify_fixed_datapath(binding);old=before['result']['resources'];new=after['result']['resources']
 assert new['LUT4']<old['LUT4']
 assert all(new[k]==old[k] for k in ('RAM10','DPRAM10','DSP24','DSP48'))
 data=json.loads((HERE/'board_resource_candidate.json').read_text())
 assert ROOT/data['selected_board']==Path(binding['reference_board'])
 data['selected_board']=board.relative_to(ROOT).as_posix()
 data['selection_reason']=(binding['variant']+': fixed RGB parcel address/return wiring and bounded counts; '
  'the parcel_quant option also folds exact Q17 correction into the existing DSP P+C stage. '
  f'Whole125-source map {new["LUT4"]}LUT4/{new["FF"]}FF, RAM250/DSP160 unchanged. '
  'Three-seed real-RAM DMA is cycle-identical; four-state arithmetic and native cluster/active cancel tests are separately bound. '
  'No new full-frame, fit, STA or board approval.')
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

