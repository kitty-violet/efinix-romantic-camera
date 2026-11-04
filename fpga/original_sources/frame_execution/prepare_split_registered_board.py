"""Freeze exactly two reviewed datapath changes into the full zero-COMB4 board."""
import json,shutil,xml.etree.ElementTree as ET
from pathlib import Path
from run_units import ROOT,HERE,sha
from split_logic_outputs_board import audit_delta

BASE=ROOT/'tmp/frame-execution/connected-board-split-outputs1'
BOARD=ROOT/'tmp/frame-execution/connected-board-split-registered2'
VAR=HERE/'variants/fixed_split_registered_edges'
GATES=ROOT/'tmp/frame-execution/split-registered-edges-shorts4'
RATE=ROOT/'tmp/frame-execution/registered-source-rate2'
NAMES=('fixed_rq32_mixed.sv','fixed_source_columns6.sv')

def main():
 assert not BOARD.exists();audit_delta(BASE)
 assert sha(BASE/'inputs.json')=='5fe31a1a300542b6f068bd3a6cd4805df97bb126deaed9bb4f1d4c599b8849cc'
 gates=json.loads((GATES/'result.json').read_text())
 assert gates['status']=='RQ_CYCLE_EQUIVALENT_AND_SOURCE_NUMERIC_SHORTS_PASS'
 for kind in ('rq','source'):
  records=json.loads((GATES/kind/'inputs.json').read_text())
  for item in records['inputs']:
   assert sha(item['source'])==item['source_sha256']
   assert sha(item['snapshot'])==item['sha256']
  result=json.loads((GATES/kind/'result.json').read_text())
  for i,case in enumerate(result['results']):assert sha(GATES/kind/f'simulation{i}.log')==case['log_sha256']
 for name in NAMES:
  kind='rq' if name==NAMES[0] else 'source'
  assert sha(VAR/name)==sha(GATES/kind/name)
 rate=json.loads((RATE/'result.json').read_text())
 assert rate['status']=='REGISTERED_SOURCE_CONTINUOUS_RATE_PASS'
 assert sha(VAR/NAMES[1])==sha(RATE/NAMES[1])
 meta=json.loads((BASE/'inputs.json').read_text())
 # The inherited advisory source_files list still points to an old ancestor.
 # Use the actual synthesis XML and the SHA-bound manifest, never those paths.
 manifest={Path(r['snapshot']).relative_to(BASE).as_posix():r for r in meta['inputs']}
 source_rels=[n.attrib['name'] for n in ET.parse(BASE/'ti60f225_oob.xml').iter()
              if n.tag.endswith('}design_file')]
 assert len(source_rels)==len(set(source_rels))
 assert all(r in manifest for r in source_rels)
 BOARD.mkdir()
 for folder in ('rtl','assembly'):shutil.copytree(BASE/folder,BOARD/folder)
 records=[]
 for rel,row in manifest.items():
  old=BASE/rel;assert sha(old)==row['sha256']
  src=VAR/old.name if rel in {'assembly/'+n for n in NAMES} else old
  dst=BOARD/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
  records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst),base_sha256=row['sha256']))
 meta.update(inputs=records,base=str(BASE),base_inputs_sha256=sha(BASE/'inputs.json'),
  source_files=[str(BOARD/r) for r in source_rels],
  registered_boundary_transform_sha256=sha(__file__),
  overrides=[dict(path=str(VAR/n),sha256=sha(VAR/n)) for n in NAMES],
  board_release=False,board_download_allowed=False,hardware_access=False)
 (BOARD/'inputs.json').write_text(json.dumps(meta,indent=2))
 changed=[]
 for item in meta['inputs']:
  dest=Path(item['snapshot']);relative=dest.relative_to(BOARD)
  assert sha(dest)==item['sha256']
  if sha(dest)!=sha(BASE/relative):changed.append(relative.as_posix())
 assert set(changed)=={'assembly/'+n for n in NAMES}
 for folder in ('rtl','assembly'):
  for old in (BASE/folder).rglob('*'):
   if old.is_file():
    rel=old.relative_to(BASE)
    assert sha(BOARD/rel)==(sha(VAR/old.name) if rel.as_posix() in changed else sha(old))
 record=dict(status='FULL_BOARD_REGISTERED_BOUNDARIES_FROZEN_NOT_RELEASED',
  board=str(BOARD),inputs_sha256=sha(BOARD/'inputs.json'),parent_inputs_sha256=sha(BASE/'inputs.json'),
  changed=changed,short_gates=dict(path=str(GATES),sha256=sha(GATES/'result.json')),
  rate_gate=dict(path=str(RATE),sha256=sha(RATE/'result.json')),
  public_rq_cycle_equivalence=True,source_initiation_interval_cycles=1,
  source_first_token_after_command_cycles=3,
  pins_clocks_constraints_protected_ip_unchanged=True,full_video_a_preserved=True,
  map_complete=False,adjacent_full_compute_integration_complete=False,
  pnr=False,board_release=False,hardware_access=False)
 (BOARD/'boundary_binding.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)

if __name__=='__main__':main()
