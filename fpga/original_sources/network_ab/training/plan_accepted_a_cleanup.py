"""Read-only deletion plan for explicitly authorized training-owned cleanup."""
import collections, json
from pathlib import Path

W=Path(r'E:\fpga\model-training-workspace')
M=W/'runs/b1-minimum-area-20261006'


def main():
    audit=W/'cleanup-20261008';inventory=json.loads((audit/'inventory.json').read_text('utf-8'))
    protected=set();origins={};known={r['path'].lower() for r in inventory['files']}
    def keep(p,why):
        p=Path(p)
        if str(p).lower() in known:protected.add(str(p).lower());origins[str(p)]=why
    def refs(obj,why):
        if isinstance(obj,dict):
            for v in obj.values():refs(v,why)
        elif isinstance(obj,list):
            for v in obj:refs(v,why)
        elif isinstance(obj,str) and len(obj)>3 and obj[1]==':' and str(Path(obj)).lower() in known:keep(obj,why)
    for name in (M/'native-luma-labels-v1/data.json',W/'runs/direction-review-20261003/freeze.json',W/'runs/style-library/black-white-line-v1-20261004/data.json'):
        refs(json.loads(name.read_text('utf-8')),str(name));keep(name,'retained manifest')
    for p in (M/'native-plain-blocks/candidate_000900.pt',W/'runs/b1-late-contour-context-20261005-v1/step_002200/coefficients-padded.bin',W/'runs/texture-recovery-20261003/final/refined_style_fp32.pt'):
        assert p.is_file(),p;keep(p,'accepted A constructor/reproduction or preserved teacher')
    trees=[M/'user-three-revisions-20261008',M/'joint-colour-levels-palette-calibration/calibrated',M/'native-boundary-field-probe/span5',W/'runs/style-library',W/'runs/direction-review-20261003',W/'runs/clear-style-control-20261003/video_sources',W/'data/processed/style-target-20261001-v14-expanded-reviewed']
    text={'.py','.ps1','.md','.txt','.json','.log','.csv','.yaml','.yml','.toml','.ini'}
    deletions=[];kept=collections.Counter();reasons=collections.Counter()
    for r in inventory['files']:
        p=Path(r['path']);reason=None
        if str(p).lower() in protected or any(p.is_relative_to(q) for q in trees):kept['accepted_and_preserved_dependencies']+=r['bytes'];continue
        if p.is_relative_to(W/'data/incoming'):kept['original_uploads']+=r['bytes'];continue
        if p.is_relative_to(W/'env') or p.is_relative_to(W/'env-probes'):kept['working_environments']+=r['bytes'];continue
        if p.is_relative_to(W/'cache/torch'):kept['VGG_and_torch_references']+=r['bytes'];continue
        if p.is_relative_to(W/'cache'):reason='obsolete_download_or_compilation_cache'
        elif p.is_relative_to(W/'tmp'):reason='completed_training_temporary_file'
        elif p.is_relative_to(W/'runs') or p.is_relative_to(W/'data/processed'):
            if p.suffix.lower() in text:kept['research_reports_and_source']+=r['bytes'];continue
            reason='superseded_experiment_generated_asset_or_checkpoint'
        elif p.parent==W and p.suffix.lower() in {'.mp4','.png','.jpg','.jpeg'}:reason='superseded_root_preview_copy'
        elif p.is_relative_to(Path(r'E:\fpga\network_ab\training')) and '__pycache__' in p.parts:reason='training_python_cache'
        if reason:deletions.append(dict(**r,reason=reason));reasons[reason]+=r['bytes']
        else:kept['other_small_sources']+=r['bytes']
    for r in deletions:
        p=Path(r['path'])
        assert p.is_relative_to(W) or p.is_relative_to(Path(r'E:\fpga\network_ab\training'))
        assert not p.is_relative_to(W/'data/incoming') and not p.is_relative_to(W/'accepted')
    result=dict(authorization='2026-10-08 user accepts A and requests deletion of unused network files',roots=[str(W),r'E:\fpga\network_ab\training'],inventory=str(audit/'inventory.json'),delete=deletions,delete_bytes=sum(r['bytes'] for r in deletions),keep_categories=dict(kept),reasons=dict(reasons),protected_files=origins,protected_trees=[str(p) for p in trees],keep_all_user_uploads=True,keep_hardware_and_other_streams=True)
    (audit/'plan.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('delete','protected_files')},ensure_ascii=False,indent=2))
    print('DELETE_FILES',len(deletions),'GIB',result['delete_bytes']/2**30,'EXPLICIT_REFS',len(protected))


if __name__=='__main__':main()
