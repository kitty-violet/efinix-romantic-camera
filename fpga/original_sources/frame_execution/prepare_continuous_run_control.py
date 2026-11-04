"""Derive a bounded repeatable preview controller from the verified ROM loader.

No hardware access. The existing frozen board and loader remain untouched.
"""
import json,re
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_command_repair import once

OUT=HERE/'variants/continuous_run'
BASE=ROOT/'tmp/frame-execution/connected-board-cnn-sparse-control1'

def generate():
    text=(BASE/'assembly/EfxSapphireSoc.v').read_text()
    matches=re.findall(r'^module cnn_boot_sequencer\b.*?^endmodule',text,re.M|re.S)
    assert len(matches)==1
    src=matches[0]+'\n'
    ref=(HERE/'variants/fixed_boot_rom/cnn_boot_sequencer.sv').read_text()
    ref='module cnn_boot_sequencer'+ref.split('module cnn_boot_sequencer',1)[1]
    ref=once(ref,'output wire [31:0] observed_crc\n',
             'output wire [31:0] observed_crc,\n output wire [5:0] observer_state\n')
    ref=once(ref,' assign observed_crc=~crc;',' assign observed_crc=~crc;\n assign observer_state=state;')
    assert src==ref
    # 512 successive frames are about 30 seconds at the measured single-frame
    # rate. 60 seconds is a hard run watchdog, not an assumed performance claim.
    out=once(src,'parameter integer RUN_TIMEOUT_CYCLES=750000000',
             "parameter [63:0] RUN_TIMEOUT_CYCLES=64'd9000000000")
    out=once(out,'reg [4:0] frames;','reg [9:0] frames;')
    out=once(out,'cmd_valid=!reset && !terminal && !waiting',
             'cmd_valid=!reset && (!terminal || state==WAIT_CMD) && !waiting')
    out=once(out,'begin failed<=1;fault<=code;run_allowed<=0;state<=FAIL_STOP;end',
             'begin terminal<=0;failed<=1;fault<=code;run_allowed<=0;state<=FAIL_STOP;end')
    out=once(out,'end else if(!terminal)begin','end else if(!terminal || state==WAIT_CMD)begin')
    out=once(out,"WAIT_CMD:if(rsp_data!=0)begin if(rsp_data!=32'h434e4e31)reject(1);else state<=CLEAR_CMD;end",
             "WAIT_CMD:if(rsp_data!=0)begin\n"
             "     if(rsp_data!=32'h434e4e31)reject(1);\n"
             "     else begin terminal<=0;state<=CLEAR_CMD;end\n"
             "    end")
    out=once(out,'GET_FRAMES:if(rsp_data<1 || rsp_data>16)reject(1);else begin frames<=rsp_data[4:0];state<=STATUS_LOAD;end',
             'GET_FRAMES:if(rsp_data<1 || rsp_data>512)reject(1);else begin frames<=rsp_data[9:0];state<=STATUS_LOAD;end')
    out=once(out,"{27'd0,frames}","{22'd0,frames}")
    out=once(out,'STATUS_DONE:begin terminal<=1;state<=HALT;end',
             '// Successful stop remains observable as terminal=1/phase=5, but\n'
             '    // accepts a fresh explicit command. Faults still latch HALT.\n'
             '    STATUS_DONE:begin terminal<=1;state<=WAIT_CMD;end')
    header='// Candidate: bounded continuous preview, 1..512 frames, 60s hard watchdog.\n'
    header+='// Same load/CRC/tag/program checks. Only successful completion can rearm.\n'
    return header+out

def main():
    assert not OUT.exists()
    OUT.mkdir()
    (OUT/'cnn_boot_sequencer.sv').write_text(generate(),encoding='utf8')
    old=HERE/'variants/run_budget/cnn_frame_admission.sv'
    text=old.read_text().replace('[4:0]','[9:0]').replace('<=16','<=512')
    (OUT/old.name).write_text(text,encoding='utf8')
    evidence={str(f):sha(f) for f in (Path(__file__),BASE/'assembly/EfxSapphireSoc.v',old)}
    (OUT/'derivation.json').write_text(json.dumps(dict(
        status='CONTINUOUS_PREVIEW_CONTROL_SOURCE_ONLY',max_frames=512,
        default_run_watchdog_cycles=9000000000,clock_hz=150000000,
        rearm='fresh explicit START only after successful stop; full resident verification repeats',
        fault_rearm=False,hardware_access=False,board_release=False,inputs=evidence,
        generated={p.name:sha(p) for p in OUT.glob('*.sv')}),indent=2))
    print('CONTINUOUS_CONTROL_SOURCE_PREPARED',flush=True)

if __name__=='__main__':main()
