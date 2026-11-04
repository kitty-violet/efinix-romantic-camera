"""Assemble measured results and native comparison panels without running a model."""
import json
from pathlib import Path
from statistics import mean
from PIL import Image, ImageDraw, ImageFont
from .runtime import configure_workspace, file_hash, write_json


def main():
    root=configure_workspace();run=root/'runs/clear-style-control-20261003'
    out=run/'review';out.mkdir(exist_ok=True)
    rows=[json.loads(s) for s in (run/'student_train/metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['step'] for r in rows]==list(range(1,2001)), 'Missing or repeated steps'
    stages=[]
    for a,b in [(1,100),(401,500),(901,1000),(1401,1500),(1901,2000)]:
        section=rows[a-1:b]
        stages.append({'first':a,'last':b,**{k:mean(r[k] for r in section) for k in ['loss','paired','temporal_style','seconds']}})
    stages_report={'steps':len(rows),'measured_update_seconds':sum(r['seconds'] for r in rows),
                   'max_allocated_MiB':max(r['peak_MiB'] for r in rows),'stages':stages,
                   'note':'Update times exclude validation/checkpoint writes and pauses; losses use different sampled crops, not aesthetic scores.'}
    write_json(out/'training_summary.json',stages_report)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
    for case in ['quiet_stairs','plaza_stone','park_people']:
        paths=[run/'anime'/case/'source.png',run/'guard_review'/(case+'.png'),run/'student_train/validation_002000'/(case+'.png')]
        board=Image.new('RGB',(1280,3*760),(22,26,32));d=ImageDraw.Draw(board)
        for i,(label,p) in enumerate(zip(['原图 · 原生720p','修正后的动画教师 · 原生720p','本轮2000步CNN · 原生720p'],paths)):
            im=Image.open(p).convert('RGB');assert im.size==(1280,720)
            d.text((16,i*760+7),label,font=font,fill='white');board.paste(im,(0,i*760+40))
        board.save(out/(case+'_native_comparison.png'))
        thumb=board.resize((640,1140),Image.Resampling.LANCZOS);thumb.save(out/(case+'_overview.jpg'),quality=94)
        # Native pixel crops, no resampling: surface, foliage, and faces/clothing.
        box={'quiet_stairs':(240,400,640,680),'plaza_stone':(260,440,660,720),'park_people':(0,320,400,600)}[case]
        crops=Image.new('RGB',(1200,640),(22,26,32));draw=ImageDraw.Draw(crops)
        for i,(label,p) in enumerate(zip(['原图','动画教师','2000步CNN'],paths)):
            patch=Image.open(p).convert('RGB').crop(box)
            draw.text((i*400+8,5),label,font=font,fill='white');crops.paste(patch,(i*400,40))
            draw.text((i*400+8,325),'灰度观察（非材质指标）',font=font,fill='white');crops.paste(patch.convert('L').convert('RGB'),(i*400,360))
        crops.save(out/(case+'_native_detail.png'))
    reports={k:json.loads((run/(k+'_videos/quality_review.json')).read_text(encoding='utf-8')) for k in ['teacher','guarded','student']}
    video=[]
    student=json.loads((run/'student_videos/report.json').read_text(encoding='utf-8'))
    for rec in student['records']:
        ids=rec['id'];entry={'id':ids,'video':rec['output'],'video_sha256':file_hash(rec['output']),
                           'frames':rec['validation']['frames'],'seconds':rec['validation']['seconds'],
                           'inference_seconds_mean':rec['mean_seconds'],'diagnostics':{}}
        for kind,report in reports.items():
            q=next(r for r in report['records'] if r['id']==ids)
            entry['diagnostics'][kind]={'median':q['median'],'p95':q['p95'],
                       'encoding_psnr_db_min':min(v['psnr_db'] for v in q['encoding_samples']),
                       'encoding_mae_max':max(v['mae'] for v in q['encoding_samples'])}
        video.append(entry)
    write_json(out/'comparison_summary.json',{'videos':video,'training':stages_report,
               'limitations':'Temporal diagnostics do not establish absence of flicker. All three sources excluded from gradient training but already reviewed; not blind evaluation.'})
    print(json.dumps({'training':stages_report,'videos':video},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
