"""Emit a selection proposal only after same-source resource and cycle checks."""
import argparse,json
from pathlib import Path
from run_units import ROOT,HERE,sha
from audit_radix3_dispatch import verify_radix3_binding

def rec(path):return dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path))

def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);a=p.parse_args()
 board=a.board.resolve();binding=json.loads((board/'gate_binding.json').read_text())
 before,after=verify_radix3_binding(binding);r0=before['result']['resources'];r1=after['result']['resources']
 assert r1['LUT4']<r0['LUT4']
 assert all(r1.get(k,0)<=r0.get(k,0) for k in ('RAM10','DPRAM10','DSP24','DSP48'))
 data=json.loads((HERE/'board_resource_candidate.json').read_text())
 assert (ROOT/data['selected_board']).resolve()==Path(binding['reference_board'])
 data['selected_board']=board.relative_to(ROOT).as_posix()
 change={'radix3':'Nine-phase DW/stride permutation uses two fixed three-way stages.',
  'shared_history':'PW/RGB byte delays share column history; nine-phase route is also changed.',
  'shared_existing_taps':'PW/RGB byte delays reuse only history taps already exposed by DW/stride; original nine-word route retained.'}[binding.get('variant','radix3')]
 data['selection_reason']=(change+' '+
  f"Whole 125-source main-chain map {r1['LUT4']} LUT4 ({r1['LUT4']-r0['LUT4']:+d}), "
  f"{r1['FF']} FF ({r1['FF']-r0['FF']:+d}), {r1['SRL8']} SRL8 ({r1['SRL8']-r0['SRL8']:+d}); RAM/DSP not increased. "
  "Forty four-state dispatch cases and all 24 real-RAM block cycles unchanged; 70336 numeric values, "
  "held tokens, five abort/direction-fault stages and active DMA/compute cancel-recovery pass. "
  "Fixed SRAM per-port owner contract is inherited. Resource fit, full-frame timing, STA and board approval remain false.")
 data['selected_inputs']=rec(board/'inputs.json');data['selected_result']=rec(board/'result.json');data['source_binding']=rec(board/'gate_binding.json')
 overrides={Path(r['path']).name:r for r in data['overrides']}
 for row in json.loads((board/'inputs.json').read_text())['inputs']:
  if Path(row['snapshot']).name in binding['changed_sources']:overrides[Path(row['snapshot']).name]=rec(Path(row['source']))
 data['overrides']=list(overrides.values());seen={r['path'] for r in data['gates']}
 for g in binding['gates']:
  root=Path(g['path']);r=rec(root/'result.json')
  if r['path'] in seen:continue
  result=json.loads((root/'result.json').read_text());r['status']=result['status']
  if (root/'native_inputs.json').is_file():r['input_manifest']=(root/'native_inputs.json').relative_to(ROOT).as_posix()
  data['gates'].append(r);seen.add(r['path'])
 print(json.dumps(data,indent=2))

if __name__=='__main__':main()
