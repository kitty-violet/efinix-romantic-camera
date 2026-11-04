"""Compare every synthesis warning/error with the released64 provenance.

Only project-root spelling is normalized; warning codes, line locations and
messages must match. This is an incremental warning review, not a zero-warning
claim or permission to download.
"""
import argparse, collections, json, re
from pathlib import Path
from run_units import ROOT, sha


def messages(board):
    path = board/'outflow/ti60f225_oob.map.out'
    rows = []
    for line in path.read_text(encoding='utf-8', errors='strict').splitlines():
        if re.match(r'^\s*(WARNING|ERROR|FATAL)\s*:', line):
            rows.append(line.replace(str(board), '<BOARD>').replace(board.as_posix(), '<BOARD>'))
    assert rows, 'An empty/unreadable warning report is not evidence'
    return path, collections.Counter(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--board', type=Path, required=True)
    a = p.parse_args()
    board = a.board.resolve()
    parent = ROOT/'tmp/frame-execution/connected-board-video-service-probe55-joint-control'
    manifest = ROOT/'output/deployment/board-full-150-video-service64-clock-20261017/manifest.json'
    proof = board/'color-map-audit.json'
    delta = json.loads(proof.read_text())
    m = json.loads(manifest.read_text(encoding='utf-8'))
    assert m['status'] == 'VIDEO_SERVICE_OFFLINE_RELEASED_FOR_SUPERVISED_TEST'
    assert Path(m['board']) == parent
    assert delta['status'] == 'COLOR_EXPOSURE_CURRENT_MAP_COMPOSITION_PASS'
    assert delta['map_sha256'] == sha(board/'outflow/ti60f225_oob.map.v')
    for f, h in delta['evidence'].items():
        assert sha(f) == h, f
    old_path, old = messages(parent)
    new_path, new = messages(board)
    added, removed = new-old, old-new
    result = dict(status='COLOR_WARNINGS_IDENTICAL_TO_RELEASED_PARENT' if not added and not removed else 'COLOR_WARNINGS_REQUIRE_REVIEW',
                  parent_warning_count=sum(old.values()), current_warning_count=sum(new.values()),
                  added=dict(added), removed=dict(removed), zero_warnings_claim=False,
                  board_release=False, hardware_access=False,
                  evidence={str(f):sha(f) for f in (manifest,proof,old_path,new_path,Path(__file__))})
    dest = board/'color-warnings-audit.json'
    assert not dest.exists(), 'Do not overwrite evidence'
    dest.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='evidence'}))
    if added or removed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
