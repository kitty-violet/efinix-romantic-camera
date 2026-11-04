"""Print a source-bound selection proposal; caller installs it with apply_patch."""
import argparse,json
from pathlib import Path
from run_units import ROOT,HERE,sha
from audit_compact_result_path import verify_compact_binding


def rec(path):return dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path))


def main():
    p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--reason',required=True);a=p.parse_args()
    board=a.board.resolve();binding=json.loads((board/'gate_binding.json').read_text());verify_compact_binding(binding)
    profile=HERE/'board_resource_candidate.json';data=json.loads(profile.read_text())
    assert data['selected_board']=='tmp/frame-execution/connected-board-unused-peripherals1'
    data['selected_board']=board.relative_to(ROOT).as_posix();data['selection_reason']=a.reason
    data['selected_inputs']=rec(board/'inputs.json');data['selected_result']=rec(board/'result.json')
    data['source_binding']=rec(board/'gate_binding.json')
    overrides={Path(x['path']).name:x for x in data['overrides']}
    for r in json.loads((board/'inputs.json').read_text())['inputs']:
        name=Path(r['snapshot']).name
        if name in binding['changed_sources']:overrides[name]=rec(Path(r['source']))
    data['overrides']=list(overrides.values())
    seen={g['path'] for g in data['gates']}
    for g in binding['gates']:
        root=Path(g['path']);path=root/'result.json';r=rec(path)
        if r['path'] in seen:continue
        result=json.loads(path.read_text());r['status']=result['status']
        if (root/'native_inputs.json').is_file():r['input_manifest']=(root/'native_inputs.json').relative_to(ROOT).as_posix()
        data['gates'].append(r);seen.add(r['path'])
    print(json.dumps(data,indent=2))


if __name__=='__main__':main()
