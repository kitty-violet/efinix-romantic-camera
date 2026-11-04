"""Cache the three RQ queue heads without changing credits or public cycles."""
import argparse,json,shutil
from pathlib import Path
from run_units import ROOT,sha
from local_control_decode import audit_delta as audit_parent
from control_command_repair import once

BASE=ROOT/'tmp/frame-execution/connected-board-local-decode1'
REL='assembly/fixed_rq32_mixed.sv'

def transform(text):
    text=once(text,'  wire [265:0] front;wire [2:0] address=3\'d0-stored[s][2:0];',
        '  // The SRL retains every queued entry. Read the second-oldest entry\n'
        '  // before a pop and cache it at that same edge. The public head is\n'
        '  // now a register, not a count-addressed mux on the feedback path.\n'
        '  reg [265:0] head;\n'
        '  wire [265:0] front;wire [2:0] address=3\'d1-stored[s][2:0];')
    text=once(text,'out_data[s*256+:256]}=front;', 'out_data[s*256+:256]}=head;')
    text=once(text,'  assign queue_bad[s]=',
        '  // No reset on wide data: stored/active retain the original reset\n'
        '  // qualification, and the first push always writes the head.\n'
        '  always @(posedge clk)if(active)begin\n'
        '   if(push && (stored[s]==0 || (stored[s]==1 && pop)))head<=payload;\n'
        '   else if(pop && stored[s]>1)head<=front;\n'
        '  end\n'
        '  assign queue_bad[s]=')
    return text

def audit_delta(board):
    audit_parent(BASE)
    meta=json.loads((board/'inputs.json').read_text());changed=[]
    assert meta['parent']==str(BASE) and meta['parent_inputs_sha256']==sha(BASE/'inputs.json')
    assert meta['cached_head_transform_sha256']==sha(__file__)
    parent=json.loads((BASE/'inputs.json').read_text())
    expected={Path(r['snapshot']).relative_to(BASE).as_posix() for r in parent['inputs']}
    actual=set()
    for row in meta['inputs']:
        dst=Path(row['snapshot']);assert dst.is_relative_to(board) and sha(dst)==row['sha256']
        rel=dst.relative_to(board).as_posix();actual.add(rel)
        if rel==REL:
            assert dst.read_text()==transform((BASE/rel).read_text());changed.append(rel)
        else:assert sha(dst)==sha(BASE/rel),rel
    assert actual==expected and changed==[REL]
    return dict(status='RQ_CACHED_HEAD_ONLY_DELTA',changed=changed,queue_capacity=8,
        head_register_bits=798,public_cycles_intended_unchanged=True,
        full_video_a_chain_preserved=True,pins_clocks_constraints_unchanged=True,
        hardware_access=False,board_release=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    audit_parent(BASE);root.mkdir();meta=json.loads((BASE/'inputs.json').read_text())
    for row in meta['inputs']:
        src=Path(row['snapshot']);assert src.is_relative_to(BASE) and sha(src)==row['sha256']
        rel=src.relative_to(BASE);dst=root/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
        if rel.as_posix()==REL:dst.write_text(transform(src.read_text()),encoding='utf8')
        row.update(snapshot=str(dst),sha256=sha(dst))
    meta.update(parent=str(BASE),parent_inputs_sha256=sha(BASE/'inputs.json'),
        cached_head_transform_sha256=sha(__file__),warning_reference_base=str(BASE),
        runner_sha256=sha(__file__),hardware_access=False,board_download_allowed=False)
    (root/'runner.py').write_bytes(Path(__file__).read_bytes())
    (root/'inputs.json').write_text(json.dumps(meta,indent=2))
    (root/'delta_audit.json').write_text(json.dumps(audit_delta(root),indent=2))
    print('RQ_CACHED_HEAD_BOARD_PREPARED_NOT_RELEASED')
if __name__=='__main__':main()
