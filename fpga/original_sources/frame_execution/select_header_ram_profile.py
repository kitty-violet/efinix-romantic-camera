"""Generate a profile proposal after the shared parameter RAM source and gate audit."""
import argparse,json
from pathlib import Path
from run_units import ROOT,HERE,sha
from audit_header_ram import verify_header_ram
def rec(p):return dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p))
def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);a=p.parse_args()
 board=a.board.resolve();binding=json.loads((board/'gate_binding.json').read_text())
 before,after=verify_header_ram(binding);old=before['result']['resources'];new=after['result']['resources']
 assert new['LUT4']<old['LUT4']
 assert new['RAM10']==old['RAM10']+3
 assert all(new[k]==old[k] for k in ('DPRAM10','DSP24','DSP48'))
 assert new['RAM10']+new['DPRAM10']<=256
 data=json.loads((HERE/'board_resource_candidate.json').read_text())
 assert ROOT/data['selected_board']==Path(binding['reference_board'])
 data['selected_board']=board.relative_to(ROOT).as_posix()
 data['selection_reason']=(f'Two-layer parameter headers in 3 real RAMs with a two-group cache: complete125-source map {new["LUT4"]}LUT4/{new["FF"]}FF. '
  'Four changed RTL sources bind shared resident header/weight reads, simultaneous next-bank writes, unchanged 24 compute block cycles, and native/four-state active cancellation. '
  'RAM253/DSP160; header response adds one transport clock. Full-frame throughput, physical capacity/timing and board approval remain open.')
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

