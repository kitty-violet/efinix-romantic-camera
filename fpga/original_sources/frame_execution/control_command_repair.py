"""Two real command CDC flops and coherent live+first-fault observation only."""
import json
from pathlib import Path
from run_units import ROOT,HERE,sha
from prepare_ddr_shift_guard_board import audit_delta as audit_parent
from variants.command_cdc_registers.transform import transform as preserve_command

BASE=ROOT/'tmp/frame-execution/connected-board-ddr-shift-guard2'

def once(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new,1)

def soc(text):
    text=preserve_command(text)
    text=once(text,'input wire [191:0] diagnostic_payload','input wire [319:0] diagnostic_payload')
    text=once(text,' wire [191:0] observation_payload;',
              ' wire [319:0] observation_payload;\n wire [191:0] first_fault_payload;')
    text=once(text,'.boot_failed(boot_failed),.record_data(observation_payload));',
              '.boot_failed(boot_failed),.record_data(first_fault_payload));\n'
              ' // Low 192 bits preserve the first-fault record; upper 128 show current progress.\n'
              ' assign observation_payload={observation_ticks,\n'
              '  observation_replies,observation_accepts,observation_last,observation_bits,\n'
              "  3'd0,bus_age,boot_observer_state,state_oh,observation_flags,first_fault_payload};")
    return text

def snapshot(text):
    assert text.count('[191:0]')==3 and text.count('[255:0]')==1
    text=text.replace('[191:0]','[319:0]').replace('[255:0]','[383:0]')
    text=once(text,"8'd1,4'd0","8'd2,4'd0")
    text=once(text,"32'h4f465331","32'h4f465032")
    text=once(text,"shifter[255:1]","shifter[383:1]")
    return text

def audit_delta(board):
    audit_parent(BASE)
    meta=json.loads((board/'inputs.json').read_text())
    assert meta['parent']==str(BASE) and meta['parent_inputs_sha256']==sha(BASE/'inputs.json')
    expected={'assembly/EfxSapphireSoc.v':soc,'assembly/jtag_bringup_snapshot.sv':snapshot}
    changed=[]
    for row in meta['inputs']:
        p=Path(row['snapshot']);assert p.is_relative_to(board) and sha(p)==row['sha256']
        rel=p.relative_to(board).as_posix();old=BASE/rel
        if rel in expected:
            assert p.read_text()==expected[rel](old.read_text()),rel
            changed.append(rel)
        else:assert sha(p)==sha(old),rel
    assert set(changed)==set(expected)
    return dict(status='COMMAND_CDC_AND_LIVE_OBSERVATION_ONLY_DELTA',changed=changed,
                first_fault_bits=192,live_bits=128,scan_bits=386,
                pins_clocks_constraints_and_functional_state_machines_unchanged=True,
                command_register_stage_count_unchanged=True,full_video_a_chain_preserved=True,
                hardware_access=False,board_release=False)
