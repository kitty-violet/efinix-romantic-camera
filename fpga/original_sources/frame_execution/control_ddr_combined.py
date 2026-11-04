"""Full-board implementation workaround and independent DDR observation.

Do not rewrite a frozen parent or claim the workaround is board-proven.
"""
import argparse,json,shutil
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_command_repair import audit_delta as audit_parent,once

BASE=ROOT/'tmp/frame-execution/connected-board-command-progress1'

COUNTERS=''' // Sampled progress only; modulo-256 counters survive functional reset.
 // No observation output feeds the DDR or control state machines.
 (* async_reg="true", syn_srlstyle="registers", syn_preserve="true" *) reg [1:0] shift_meta=0,shift_sync=0;
 reg [1:0] shift_previous=0;
 reg previous_system_reset_n=0,previous_ddr_lock=0;
 reg [7:0] shift_requests=0,shift_forwarded=0,system_reset_events=0,ddr_lock_drops=0;
 always @(posedge clk) begin
  shift_meta<=diagnostic_shift_events;shift_sync<=shift_meta;shift_previous<=shift_sync;
  previous_system_reset_n<=diagnostic_board_status[0];
  previous_ddr_lock<=diagnostic_board_status[3];
  if(shift_sync[0] && !shift_previous[0])shift_requests<=shift_requests+1'b1;
  if(shift_sync[1] && !shift_previous[1])shift_forwarded<=shift_forwarded+1'b1;
  if(previous_system_reset_n && !diagnostic_board_status[0])system_reset_events<=system_reset_events+1'b1;
  if(previous_ddr_lock && !diagnostic_board_status[3])ddr_lock_drops<=ddr_lock_drops+1'b1;
 end
'''

def soc(text):
    text=once(text,'input wire [319:0] diagnostic_payload','input wire [383:0] diagnostic_payload')
    text=once(text,'input wire [31:0] diagnostic_board_status,','input wire [31:0] diagnostic_board_status,\n  input wire [1:0] diagnostic_shift_events,')
    text=once(text,'wire [319:0] observation_payload;','wire [383:0] observation_payload;')
    text=once(text,' // Low 192 bits preserve the first-fault record; upper 128 show current progress.',COUNTERS+' // Low 320 bits preserve OFP2; upper 64 add DDR status and sampled events.')
    return once(text,'assign observation_payload={observation_ticks,','assign observation_payload={ddr_lock_drops,system_reset_events,shift_forwarded,shift_requests,\n  diagnostic_board_status,observation_ticks,')

def snapshot(text):
    assert text.count('[319:0]')==3
    text=text.replace('[383:0]','[447:0]').replace('[319:0]','[383:0]')
    text=once(text,"8'd2,4'd0","8'd3,4'd0")
    text=once(text,"32'h4f465032","32'h4f465033")
    return once(text,'shifter[383:1]','shifter[447:1]')

def top(text):
    text=once(text,'wire [2:0] bringup_cal_phase;','wire [2:0] bringup_cal_phase;\nwire [35:0] independent_ddr_debug;')
    text=once(text,'.diagnostic_board_status(bringup_ddr_sync),','.diagnostic_board_status(bringup_ddr_sync),\n .diagnostic_shift_events(independent_ddr_debug[14:13]),')
    return once(text,'.rdlvl_shift                   (bringup_cal_phase),','.ddr_debug_port                (independent_ddr_debug),\n.rdlvl_shift                   (bringup_cal_phase),')

def ddr(text):
    # Two existing public interface signals only. The actual PHY is untouched.
    return once(text,"assign ddr_debug_port={23'b0,calibration_log,calibration_phase,calibration_pass,calibration_done};",
        "assign ddr_debug_port={21'b0,pll_shift_ena,request_enable,calibration_log,calibration_phase,calibration_pass,calibration_done};")

def xml(text):
    return once(text,'name="pack-luts-to-comb4" value="1"','name="pack-luts-to-comb4" value="0"')

TRANSFORMS={'assembly/EfxSapphireSoc.v':soc,'assembly/jtag_bringup_snapshot.sv':snapshot,
            'rtl/ti60f225_oob_top.v':top,'ddr516_v4_controller.sv':ddr,'ti60f225_oob.xml':xml}

def audit_delta(board):
    audit_parent(BASE)
    meta=json.loads((board/'inputs.json').read_text());changed=[]
    assert meta['parent']==str(BASE) and meta['parent_inputs_sha256']==sha(BASE/'inputs.json')
    assert meta['combined_transform_sha256']==sha(__file__)
    for row in meta['inputs']:
        dst=Path(row['snapshot']);assert dst.is_relative_to(board) and sha(dst)==row['sha256']
        rel=dst.relative_to(board).as_posix();src=BASE/rel
        if rel in TRANSFORMS:
            assert dst.read_text()==TRANSFORMS[rel](src.read_text()),rel;changed.append(rel)
        else:assert sha(dst)==sha(src),rel
    assert set(changed)==set(TRANSFORMS),(changed,list(TRANSFORMS))
    return dict(status='NO_COMB4_AND_INDEPENDENT_DDR_OBSERVATION_DELTA',changed=changed,
        functional_rtl_unchanged=True,full_video_a_chain_preserved=True,pins_clocks_constraints_unchanged=True,
        observation_format='OFP3',payload_bits=384,scan_bits=450,
        shift_counts_are_sampled_enable_rises_not_physical_pll_completion=True,
        hardware_access=False,board_release=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    audit_parent(BASE);root.mkdir();meta=json.loads((BASE/'inputs.json').read_text())
    for row in meta['inputs']:
        src=Path(row['snapshot']);assert src.is_relative_to(BASE) and sha(src)==row['sha256']
        rel=src.relative_to(BASE);dst=root/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
        if rel.as_posix() in TRANSFORMS:dst.write_text(TRANSFORMS[rel.as_posix()](src.read_text()),encoding='utf-8')
        row.update(snapshot=str(dst),sha256=sha(dst))
    meta.update(parent=str(BASE),parent_inputs_sha256=sha(BASE/'inputs.json'),warning_reference_base=str(BASE),
        combined_transform_sha256=sha(__file__),runner_sha256=sha(__file__),hardware_access=False,board_download_allowed=False)
    (root/'runner.py').write_bytes(Path(__file__).read_bytes())
    (root/'inputs.json').write_text(json.dumps(meta,indent=2))
    (root/'delta_audit.json').write_text(json.dumps(audit_delta(root),indent=2))
    print('COMBINED_FULL_BOARD_FROZEN_NOT_RELEASED')
if __name__=='__main__':main()
