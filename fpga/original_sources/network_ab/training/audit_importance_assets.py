"""Audit already prepared training assets and exact time separation."""
from collections import Counter
import json
from .importance_pruned_teacher import RUN
from .runtime import configure_workspace,file_hash,write_json


def main():
    root=configure_workspace();run=root/'runs'/RUN
    manifest=json.loads((run/'data_manifest.json').read_text(encoding='utf-8'))
    expected={};gaps=[]
    for row in manifest['pairs']:
        assert row['split']=='train' and manifest['source_splits'][row['source']]['split']=='train'
        gaps.append(row['next_pts']-row['current_pts'])
        for key in ['current','next','current_target','next_target','flow']:
            path=row[key];sha=row[key+'_sha256']
            assert path not in expected or expected[path]==sha
            expected[path]=sha
    for path,sha in expected.items():assert file_hash(path)==sha,path
    result=dict(status='PASS',files_checked=len(expected),pairs=len(gaps),
                source_count=len({r['source'] for r in manifest['pairs']}),
                frame_index_gaps=dict(Counter(str(r['next_index']-r['current_index']) for r in manifest['pairs'])),
                pts_gap_min_seconds=min(gaps),pts_gap_max_seconds=max(gaps),
                clarification='Training and ranking use nearby sampled pairs (~0.1s apart), not every consecutive 30fps frame. Delivery videos use every actual frame and temporal diagnostics cover every adjacent pair.',
                manifest_sha256=file_hash(run/'data_manifest.json'))
    write_json(run/'training_assets_audit.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
