"""Independent full-video/A clone: official no-COMB4 synthesis option.

Functional RTL and protected IP stay byte-identical. Only the two finite
observation instances are expressed as equivalent separate LUT4 outputs.
"""
import argparse,json,shutil
from pathlib import Path
from run_units import ROOT,HERE,sha
from startup_cause_board import audit_delta as audit_parent

BASE=ROOT/'tmp/frame-execution/connected-board-startup-cause4'
CHANGED={'ti60f225_oob.xml','assembly/bringup_cell_probe.sv'}
OLD=''' EFX_COMB4 #(.MODE("LOGIC"),.LUTMASK(16'h1144)) pair_under_test(
  .I0(index[0]),.I1(index[1]),.I2(1'b0),.I3(1'b0),.CI(1'b0),.O(comb_o),.P(comb_p),.CO(),.FCO());'''
NEW=''' // Same finite observation truth functions, with separate ordinary LUTs.
 EFX_LUT4 #(.LUTMASK(16'h4444)) independent_o(
  .I0(index[0]),.I1(index[1]),.O(comb_o));
 EFX_LUT4 #(.LUTMASK(16'h1111)) independent_p(
  .I0(index[0]),.I1(index[1]),.O(comb_p));'''

def transform(rel,text):
 if rel=='ti60f225_oob.xml':
  before='name="pack-luts-to-comb4" value="1"'
  assert text.count(before)==1
  return text.replace(before,'name="pack-luts-to-comb4" value="0"')
 if rel=='assembly/bringup_cell_probe.sv':
  assert text.count(OLD)==1
  return text.replace(OLD,NEW)
 return text

def truth_proof():
 for i in range(16):
  assert (0x4444>>i&1)==(0x1144>>(i&3)&1)
  assert (0x1111>>i&1)==(0x1144>>(8+(i&3))&1)
 return dict(status='SPLIT_PROBE_OUTPUT_TRUTH_PASS',checks=32,
  scope='Original probe I2/I3 tied low, both LUTs now ignore unconnected I2/I3.')

def audit_delta(board):
 audit_parent(BASE)
 meta=json.loads((board/'inputs.json').read_text());parent=json.loads((BASE/'inputs.json').read_text())
 assert meta['parent']==str(BASE) and meta['parent_inputs_sha256']==sha(BASE/'inputs.json')
 assert meta['split_logic_transform_sha256']==sha(__file__)
 expected={Path(r['snapshot']).relative_to(BASE).as_posix() for r in parent['inputs']}
 seen=set();changed=[]
 for r in meta['inputs']:
  f=Path(r['snapshot']);assert f.is_relative_to(board) and sha(f)==r['sha256']
  rel=f.relative_to(board).as_posix();seen.add(rel)
  if rel in CHANGED:
   assert f.read_text()==transform(rel,(BASE/rel).read_text()),rel;changed.append(rel)
  else:assert sha(f)==sha(BASE/rel),rel
 assert seen==expected and set(changed)==CHANGED
 return dict(status='OFFICIAL_NO_COMB4_AND_EQUIVALENT_PROBE_DELTA',changed=changed,
  functional_rtl_unchanged=True,protected_ip_unchanged=True,pins_clocks_constraints_unchanged=True,
  full_video_a_preserved=True,truth_proof=truth_proof(),hardware_access=False,
  actual_map_zero_comb4_proven=False,physical_pass=False,board_release=False)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 board=a.root.resolve();assert board.is_relative_to(ROOT/'tmp') and not board.exists()
 audit_parent(BASE);meta=json.loads((BASE/'inputs.json').read_text());board.mkdir()
 for r in meta['inputs']:
  src=Path(r['snapshot']);assert src.is_relative_to(BASE) and sha(src)==r['sha256']
  rel=src.relative_to(BASE).as_posix();dst=board/rel;dst.parent.mkdir(parents=True,exist_ok=True)
  if rel in CHANGED:dst.write_text(transform(rel,src.read_text()),encoding='utf8')
  else:shutil.copy2(src,dst)
  r.update(snapshot=str(dst),sha256=sha(dst))
 meta.update(parent=str(BASE),parent_inputs_sha256=sha(BASE/'inputs.json'),
  split_logic_transform_sha256=sha(__file__),warning_reference_base=str(BASE),
  hardware_access=False,board_download_allowed=False)
 (board/'inputs.json').write_text(json.dumps(meta,indent=2))
 (board/'delta_audit.json').write_text(json.dumps(audit_delta(board),indent=2))
 print(json.dumps(dict(status='PREPARED_SPLIT_OUTPUTS_NOT_RELEASED',board=str(board),
  inputs_sha256=sha(board/'inputs.json'),delta_sha256=sha(board/'delta_audit.json'))))

if __name__=='__main__':main()
