"""Post-run training coverage, image detail and encoding checks; no score-as-pass."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from .runtime import configure_workspace,file_hash,write_json


def edge_mean(a):
    a=a.astype(np.float32)
    return float((np.abs(np.diff(a,axis=0)).mean()+np.abs(np.diff(a,axis=1)).mean())/2)


def main():
    root=configure_workspace();run=root/'runs/b1-distillation-20261003';cv2.setNumThreads(2)
    data=json.loads((run/'data/manifest.json').read_text(encoding='utf-8'))
    logs=[json.loads(s) for s in (run/'train/metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['step'] for r in logs]==list(range(1,2001))
    assert all(np.isfinite(r['loss']) and r['grad_norm']>0 for r in logs)
    train_sources={p['source'] for p in data['pairs']}
    valid_sources={p['raw_source'] for p in data['validation']}
    assert not train_sources&valid_sources
    pre=json.loads((run/'preflight/report.json').read_text());binding=json.loads((run/'train/run.json').read_text())
    assert pre['binding']==binding['binding'] and binding['resumed_from'].endswith('checkpoint_000500.pt')
    weights=run/'final-2000/b1_style_fp32.pt';exp=json.loads((run/'final-2000/verification.json').read_text())
    assert file_hash(weights)==exp['sha256']
    comparisons=[]
    for row in data['validation']:
        x=np.array(Image.open(row['source']));t=np.array(Image.open(row['target']))
        y=np.array(Image.open(run/'train/validation_002000'/(row['id']+'.png')))
        comparisons.append({'id':row['id'],'source_edge_mean':edge_mean(x),'teacher_edge_mean':edge_mean(t),
                            'b1_edge_mean':edge_mean(y),'b1_teacher_mae':float(np.abs(y.astype(np.float32)-t).mean())})
    out=run/'videos-2000';video=json.loads((out/'report.json').read_text());assert video['status']=='COMPLETE_REQUIRES_VISUAL_REVIEW'
    stats=[];codec=[]
    for row in video['records']:
        assert row['raw_source'] not in train_sources
        temporal=lambda k:[r[k] for r in row['stats'] if r.get(k) is not None]
        st=temporal('student_temporal_style_mae');tt=temporal('teacher_temporal_style_mae')
        stats.append({'id':row['id'],'frames':row['student_validation']['frames'],'seconds':row['student_validation']['seconds'],
                      'teacher_mae':float(np.mean([r['teacher_error_rgb255'] for r in row['stats']])),
                      'student_source_change':float(np.mean([r['source_change_rgb255'] for r in row['stats']])),
                      'teacher_source_change':float(np.mean([r['teacher_source_change_rgb255'] for r in row['stats']])),
                      'temporal_samples':len(st),'student_temporal_p95':float(np.percentile(st,95)),
                      'teacher_temporal_p95':float(np.percentile(tt,95))})
        cap=cv2.VideoCapture(row['student'])
        for n in row['samples']:
            cap.set(cv2.CAP_PROP_POS_FRAMES,n);ok,bgr=cap.read();assert ok
            actual=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB).astype(np.float32)
            ref=np.array(Image.open(out/f'{row["id"]}_{n:03d}_b1_preencode1080.png')).astype(np.float32)
            mse=float(np.square(actual-ref).mean());psnr=10*np.log10(255**2/max(mse,1e-12))
            codec.append({'id':row['id'],'frame':n,'rgb_psnr_db':float(psnr)})
        cap.release()
    result={'status':'NUMERICAL_AND_DELIVERY_CHECKS_PASS_VISUAL_QUALITY_SEPARATE','actual_steps':2000,
            'native720_update_count':sum(r['native'] for r in logs),
            'sampled_training_pairs':len({(s['source'],s['index']) for r in logs for s in r['samples']}),
            'training_pairs':len(data['pairs']),'train_sources':len(train_sources),'regression_sources':len(valid_sources),
            'source_overlap':[],'optimizer_update_seconds':sum(r['seconds'] for r in logs),
            'peak_allocated_mib':max(r['peak_MiB'] for r in logs),'resume_error':pre['replay_error'],
            'weights':str(weights),'weights_sha256':file_hash(weights),'weights_bytes':weights.stat().st_size,
            'still_detail':comparisons,'videos':stats,'encoding_samples':codec,
            'quality_limit':'A low temporal residual can result from blur; it cannot alone prove flicker-free or acceptable texture/person detail.',
            'contract_file_sha256':file_hash(Path(__file__).parents[1]/'model_b_contract.json'),
            'student_postprocess':'none','qat_completed':False,'fpga_deployed':False,
            'evidence':{str(p):file_hash(p) for p in [Path(__file__),Path(__file__).with_name('render_b1_distillation.py'),run/'config.json',run/'data/manifest.json']}}
    write_json(run/'verification.json',result)
    print(json.dumps({k:result[k] for k in ['status','actual_steps','sampled_training_pairs','optimizer_update_seconds','weights_bytes','videos']},indent=2))


if __name__=='__main__':main()
