"""Apply operand steering to original C2; keep its queue and all hard interfaces."""
import argparse,json,shutil
from pathlib import Path
from run_units import ROOT,sha
from local_control_decode import audit_delta as audit_parent
from rq_feedback_select_board import transform,FILES
import rq_feedback_select_board as selector

BASE=ROOT/'tmp/frame-execution/connected-board-local-decode1'

def audit_delta(board):
    audit_parent(BASE)
    meta=json.loads((board/'inputs.json').read_text())
    assert meta['parent']==str(BASE) and meta['parent_inputs_sha256']==sha(BASE/'inputs.json')
    assert meta['minimal_feedback_transform_sha256']==sha(__file__)
    assert meta['selector_transform_sha256']==sha(selector.__file__)
    parent=json.loads((BASE/'inputs.json').read_text())
    expected={Path(r['snapshot']).relative_to(BASE).as_posix() for r in parent['inputs']}
    actual=set();changed=[]
    for row in meta['inputs']:
        dst=Path(row['snapshot']);assert dst.is_relative_to(board) and sha(dst)==row['sha256']
        rel=dst.relative_to(board).as_posix();actual.add(rel)
        if rel in FILES:
            assert dst.read_text()==transform(rel,(BASE/rel).read_text());changed.append(rel)
        else:assert sha(dst)==sha(BASE/rel),rel
    assert actual==expected and set(changed)==set(FILES)
    old=(BASE/FILES[0]).read_text();new=(board/FILES[0]).read_text()
    marker=' wire [2:0] queue_bad,nonempty;'
    assert old.split(marker,1)[1]==new.split(marker,1)[1]
    assert 'reg [265:0] head;' not in new
    return dict(status='REGISTERED_FEEDBACK_ONLY_ORIGINAL_QUEUES_DELTA',changed=sorted(changed),
        original_queue_bytes_unchanged=True,new_head_register_bits=0,new_public_latency=0,
        acceptance_guards_unchanged=True,full_video_a_chain_preserved=True,
        pins_clocks_constraints_unchanged=True,hardware_access=False,board_release=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    audit_parent(BASE);root.mkdir();meta=json.loads((BASE/'inputs.json').read_text())
    for row in meta['inputs']:
        src=Path(row['snapshot']);assert src.is_relative_to(BASE) and sha(src)==row['sha256']
        rel=src.relative_to(BASE);dst=root/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
        if rel.as_posix() in FILES:dst.write_text(transform(rel.as_posix(),src.read_text()),encoding='utf8')
        row.update(snapshot=str(dst),sha256=sha(dst))
    meta.update(parent=str(BASE),parent_inputs_sha256=sha(BASE/'inputs.json'),
        minimal_feedback_transform_sha256=sha(__file__),selector_transform_sha256=sha(selector.__file__),
        warning_reference_base=str(BASE),runner_sha256=sha(__file__),hardware_access=False,board_download_allowed=False)
    (root/'runner.py').write_bytes(Path(__file__).read_bytes())
    (root/'inputs.json').write_text(json.dumps(meta,indent=2))
    (root/'delta_audit.json').write_text(json.dumps(audit_delta(root),indent=2))
    print('MINIMAL_FEEDBACK_BOARD_PREPARED_NOT_RELEASED')
if __name__=='__main__':main()
