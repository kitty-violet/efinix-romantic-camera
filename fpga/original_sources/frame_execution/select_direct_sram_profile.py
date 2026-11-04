"""Emit a verified selection proposal; apply it only with apply_patch."""
import argparse,json
from pathlib import Path
from run_units import ROOT,HERE,sha
from audit_direct_sram_response import verify_direct_binding

def rec(path):return dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path))

def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);a=p.parse_args()
 board=a.board.resolve();binding=json.loads((board/'gate_binding.json').read_text())
 before,after=verify_direct_binding(binding);r0=before['result']['resources'];r1=after['result']['resources']
 assert r1['LUT4']<r0['LUT4']
 assert all(r1.get(k,0)<=r0.get(k,0) for k in ('RAM10','DPRAM10','DSP24','DSP48'))
 data=json.loads((HERE/'board_resource_candidate.json').read_text())
 assert (ROOT/data['selected_board']).resolve()==Path(binding['reference_board'])
 data['selected_board']=board.relative_to(ROOT).as_posix()
 data['selection_reason']=(f"Fixed local SRAM port-response contract removes internal echoed tags and duplicate expected tags; "
  f"DMA retains local 3-bit reorder slots. Whole 125-source main-chain map {r1['LUT4']} LUT4 "
  f"({r1['LUT4']-r0['LUT4']:+d}), {r1['FF']} FF ({r1['FF']-r0['FF']:+d}); RAM/DSP not increased. "
  "All 24 block cycles unchanged, 70336 numeric values and five residual abort/direction-fault stages pass; "
  "three-seed dual-real-RAM DMA cycle equivalence, real RAM collision/owner replacement, guard and focused four-state regressions bound. "
  "Internal echo-tag corruption detection deliberately no longer exists; one outstanding per physical port and retained owner are mandatory. "
  "DDR IDs, bounds and responses and coefficient integrity are unchanged. Resource fit, full-frame timing, STA and board approval remain false.")
 data['selected_inputs']=rec(board/'inputs.json');data['selected_result']=rec(board/'result.json');data['source_binding']=rec(board/'gate_binding.json')
 overrides={Path(r['path']).name:r for r in data['overrides']}
 for row in json.loads((board/'inputs.json').read_text())['inputs']:
  name=Path(row['snapshot']).name
  if name in binding['changed_sources']:overrides[name]=rec(Path(row['source']))
 data['overrides']=list(overrides.values());seen={r['path'] for r in data['gates']}
 for g in binding['gates']:
  root=Path(g['path']);r=rec(root/'result.json')
  if r['path'] in seen:continue
  result=json.loads((root/'result.json').read_text());r['status']=result['status']
  if (root/'native_inputs.json').is_file():r['input_manifest']=(root/'native_inputs.json').relative_to(ROOT).as_posix()
  data['gates'].append(r);seen.add(r['path'])
 print(json.dumps(data,indent=2))

if __name__=='__main__':main()
