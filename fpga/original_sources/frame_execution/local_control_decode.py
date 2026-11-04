"""Isolate exact ten-state decode with explicit single-output LUT primitives.

Keep full-board packing enabled. This is a local implementation alternative,
not a claim that the earlier COMB4 fault hypothesis is proven on silicon.
"""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_command_repair import audit_delta as audit_parent,once
from control_ddr_combined import soc as observe_soc,snapshot,top,ddr

BASE=ROOT/'tmp/frame-execution/connected-board-command-progress1'
STATES=('IDLE','DECODE','MAIL_READ','APB_SETUP','APB_ACCESS','AXI_WRITE','AXI_B','AXI_AR','AXI_R','RESPONSE')

def decoder():
    lines=['module control_exact_state_decode(input wire [9:0] value, output wire [9:0] matched);']
    # Retain both driver and consumers, then verify the emitted primitives.
    groups=[('lo',0,4),('mi',4,3),('hi',7,3)]
    for group,start,width in groups:
        lines.append(f' (* syn_keep="true" *) wire [{width}:0] {group};')
        for index in range(width+1):
            target=0 if index==width else 1<<index
            mask=sum(1<<v for v in range(16) if (v&((1<<width)-1))==target)
            pins=[f'.I{i}(value[{start+i}])' if i<width else f".I{i}(1'b0)" for i in range(4)]
            lines.append(f" EFX_LUT4 #(.LUTMASK(16'h{mask:04x})) {group}_{index}("+','.join(pins)+f',.O({group}[{index}]));')
    for i in range(10):
        args=[f'{g}[{i-s if s<=i<s+w else w}]' for g,s,w in groups]
        lines.append(f" EFX_LUT4 #(.LUTMASK(16'h8080)) eq_{i}(.I0({args[0]}),.I1({args[1]}),.I2({args[2]}),.I3(1'b0),.O(matched[{i}]));")
    lines.append('endmodule\n')
    return '\n'.join(lines)

def soc(text):
    text=observe_soc(text)
    start=text.index(' // Fixed one-hot control;')
    end=text.index(' // Read-only, configuration-initialized evidence',start)
    block=text[start:end]
    block=once(block,' wire [3:0] state;',
        ' (* syn_keep="true" *) wire [9:0] ctrl_state_match;\n'
        ' control_exact_state_decode ctrl_decode(.value(state_oh),.matched(ctrl_state_match));\n wire [3:0] state;')
    for i,name in enumerate(STATES):
        block=re.sub(r'\bstate_oh=='+name+r'\b',f'ctrl_state_match[{i}]',block)
        block=once(block,'    '+name+':',f'    ctrl_state_match[{i}]:')
    block=once(block,'case(state_oh)',"case(1'b1)")
    assert 'state_oh==' not in block
    return text[:start]+block+text[end:]+'\n'+decoder()

TRANSFORMS={'assembly/EfxSapphireSoc.v':soc,'assembly/jtag_bringup_snapshot.sv':snapshot,
            'rtl/ti60f225_oob_top.v':top,'ddr516_v4_controller.sv':ddr}

def audit_delta(board):
    audit_parent(BASE)
    meta=json.loads((board/'inputs.json').read_text());changed=[]
    assert meta['parent']==str(BASE) and meta['parent_inputs_sha256']==sha(BASE/'inputs.json')
    assert meta['local_decode_transform_sha256']==sha(__file__)
    for row in meta['inputs']:
        dst=Path(row['snapshot']);assert dst.is_relative_to(board) and sha(dst)==row['sha256']
        rel=dst.relative_to(board).as_posix();src=BASE/rel
        if rel in TRANSFORMS:
            assert dst.read_text()==TRANSFORMS[rel](src.read_text()),rel;changed.append(rel)
        else:assert sha(dst)==sha(src),rel
    assert set(changed)==set(TRANSFORMS)
    assert 'name="pack-luts-to-comb4" value="1"' in (board/'ti60f225_oob.xml').read_text()
    return dict(status='LOCAL_EXACT_DECODE_AND_OFP3_DELTA',changed=changed,
        single_output_decoder_luts=23,controller_latency_changed=False,
        full_video_a_chain_preserved=True,pins_clocks_constraints_unchanged=True,
        synthesis_packing_unchanged=True,observation_format='OFP3',hardware_access=False,board_release=False)

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
        local_decode_transform_sha256=sha(__file__),observation_transform_sha256=sha(HERE/'control_ddr_combined.py'),
        runner_sha256=sha(__file__),hardware_access=False,board_download_allowed=False)
    (root/'runner.py').write_bytes(Path(__file__).read_bytes())
    (root/'inputs.json').write_text(json.dumps(meta,indent=2))
    (root/'delta_audit.json').write_text(json.dumps(audit_delta(root),indent=2))
    print('LOCAL_DECODE_FULL_BOARD_FROZEN_NOT_RELEASED')
if __name__=='__main__':main()
