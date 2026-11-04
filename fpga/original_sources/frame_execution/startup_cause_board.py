"""OFP5: passive initial transition history plus finite primitive raw samples."""
import argparse,json,shutil
from pathlib import Path
from run_units import ROOT,HERE,sha
from fault_probe_board import audit_delta as audit_parent,one
BASE=ROOT/'tmp/frame-execution/connected-board-fault-probe1'
VAR=HERE/'variants/startup_cause_probe'
CHANGED={'assembly/sapphire_axi32_to_axi128.sv','assembly/ti60_ddr_four_master_fabric.sv',
 'assembly/cnn_rgb_frame_pipeline.sv','assembly/cnn_video_pipeline.sv','assembly/EfxSapphireSoc.v',
 'assembly/jtag_bringup_snapshot.sv','rtl/ti60f225_oob_top.v','ti60f225_oob.xml'}
def transform(rel,text):
 if rel=='assembly/sapphire_axi32_to_axi128.sv':
  text=one(text,'[127:0] adapter_diagnostic','[255:0] adapter_diagnostic')
  return one(text,'    assign adapter_diagnostic={probe_live,probe_first};',
   (VAR/'adapter_trace.inc').read_text()+'\n    assign adapter_diagnostic={trace_first_bad,trace_first_active,probe_live,probe_first};')
 if rel in ('assembly/ti60_ddr_four_master_fabric.sv','assembly/cnn_rgb_frame_pipeline.sv','assembly/cnn_video_pipeline.sv'):
  return one(text,'[127:0] adapter_diagnostic','[255:0] adapter_diagnostic')
 if rel=='assembly/EfxSapphireSoc.v':
  text=text.replace('[703:0] diagnostic_payload','[959:0] diagnostic_payload').replace('[703:0] observation_payload','[959:0] observation_payload')
  return one(text,'[319:0] diagnostic_extra','[575:0] diagnostic_extra')
 if rel=='assembly/jtag_bringup_snapshot.sv':
  return text.replace('[703:0]','[959:0]').replace('[767:0]','[1023:0]').replace('[767:1]','[1023:1]').replace("8'd4","8'd5").replace("32'h4f465034","32'h4f465035")
 if rel=='rtl/ti60f225_oob_top.v':
  text=one(text,'wire [127:0] adapter_diagnostic;','wire [255:0] adapter_diagnostic;\nwire [127:0] cell_diagnostic;\nbringup_cell_pair_probe cell_probe(process_clk_150,sdram_clk,w_arstn,cell_diagnostic);')
  # Preserve all OFP4 fields at their previous offsets; append new data above.
  return one(text,'{ddr_clock_diagnostic,adapter_diagnostic}',
   '{cell_diagnostic,adapter_diagnostic[255:128],ddr_clock_diagnostic,adapter_diagnostic[127:0]}')
 if rel=='ti60f225_oob.xml':
  before='        <efx:top_module name="ti60f225_oob_top" />\n'
  return one(text,before,before+'        <efx:design_file name="assembly/bringup_cell_probe.sv" version="default" library="default" />\n')
 return text
def audit_delta(board):
 audit_parent(BASE);m=json.loads((board/'inputs.json').read_text());p=json.loads((BASE/'inputs.json').read_text())
 assert m['parent']==str(BASE) and m['parent_inputs_sha256']==sha(BASE/'inputs.json')
 assert m['startup_cause_transform_sha256']==sha(__file__) and m['trace_source_sha256']==sha(VAR/'adapter_trace.inc')
 expected={Path(r['snapshot']).relative_to(BASE).as_posix() for r in p['inputs']};seen=set()
 for r in m['inputs']:
  f=Path(r['snapshot']);assert f.is_relative_to(board) and sha(f)==r['sha256']
  rel=f.relative_to(board).as_posix();seen.add(rel)
  if rel=='assembly/bringup_cell_probe.sv':assert sha(f)==sha(VAR/'bringup_cell_probe.sv')
  elif rel in CHANGED:assert f.read_text()==transform(rel,(BASE/rel).read_text()),rel
  else:assert sha(f)==sha(BASE/rel),rel
 assert seen==expected|{'assembly/bringup_cell_probe.sv'}
 return dict(status='STARTUP_CAUSE_OBSERVATION_ONLY_DELTA',functional_statements_unchanged=True,
  pins_clocks_constraints_ddr_unchanged=True,full_video_a_chain_preserved=True,
  payload_bits=960,serial_bits_with_wrapper=1026,hardware_access=False,board_release=False)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args()
 board=a.root.resolve();assert board.is_relative_to(ROOT/'tmp') and not board.exists()
 audit_parent(BASE);m=json.loads((BASE/'inputs.json').read_text());board.mkdir()
 for r in m['inputs']:
  src=Path(r['snapshot']);assert sha(src)==r['sha256'];rel=src.relative_to(BASE).as_posix()
  dst=board/rel;dst.parent.mkdir(parents=True,exist_ok=True)
  if rel in CHANGED:dst.write_text(transform(rel,src.read_text()),encoding='utf8')
  else:shutil.copy2(src,dst)
  r.update(snapshot=str(dst),sha256=sha(dst))
 src=VAR/'bringup_cell_probe.sv';dst=board/'assembly'/src.name;shutil.copy2(src,dst)
 m['inputs'].append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
 m.update(parent=str(BASE),parent_inputs_sha256=sha(BASE/'inputs.json'),startup_cause_transform_sha256=sha(__file__),
  trace_source_sha256=sha(VAR/'adapter_trace.inc'),warning_reference_base=str(BASE),hardware_access=False,board_download_allowed=False)
 (board/'inputs.json').write_text(json.dumps(m,indent=2));(board/'delta_audit.json').write_text(json.dumps(audit_delta(board),indent=2))
 print('PREPARED_STARTUP_CAUSE_OBSERVATION_NOT_RELEASED',board)
if __name__=='__main__':main()
