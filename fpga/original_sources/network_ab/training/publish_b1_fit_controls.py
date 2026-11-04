"""Verify, expose native diagnostic comparisons, and render honest single-patch trials."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from network_ab.contract_torch import build_fp32_model
from .b1_fit_controls import RUN, CASES, score
from .fpga_video_resize import area_3to2, bilinear_2to3
from .render_temporal_native_detail import encode_pipe
from .render_wavelet_game import close_pipe, video_check
from .runtime import configure_workspace, contract_guard, file_hash, write_json
from .train_b1_distillation import as_tensor, to_rgb


def panel(path, entries):
    images=[np.array(Image.open(p).convert('RGB')) for _,p in entries]
    h,w=images[0].shape[:2]
    assert all(a.shape==images[0].shape for a in images)
    board=Image.new('RGB',(w*len(entries),h+28),'#111111');d=ImageDraw.Draw(board)
    for i,((label,_),a) in enumerate(zip(entries,images)):
        board.paste(Image.fromarray(a),(i*w,28));d.text((i*w+7,7),label,fill='white')
    board.save(path)


def load_net(path):
    s=torch.load(path,map_location='cpu',weights_only=True)
    assert s['contract']==contract_guard()[1]['contract_sha256']
    net=build_fp32_model().cuda().eval();net.load_state_dict(s['model']);return net


def verify(run):
    cfg=json.loads((run/'config.json').read_text(encoding='utf-8'))
    result=json.loads((run/'results.json').read_text());assert len(result['records'])==4
    checks=[]
    for r in result['records']:
        rows=[json.loads(s) for s in (run/r['case']/'metrics.jsonl').read_text().splitlines()]
        assert [v['step'] for v in rows]==list(range(1,cfg['steps']+1))
        assert all(np.isfinite(v['loss']) and v['grad_norm']>0 for v in rows)
        assert file_hash(r['weights'])==r['sha256'] and r['reload_error']==0
        checks.append({'case':r['case'],'steps':len(rows),'start_loss':rows[0]['loss'],
                       'last_loss':rows[-1]['loss'],'first100_loss':float(np.mean([s['loss'] for s in rows[:100]])),
                       'last100_loss':float(np.mean([s['loss'] for s in rows[-100:]])),
                       'weight_sha256':r['sha256']})
    for p,sha in cfg['code'].items():assert file_hash(p)==sha,p
    for task in ('identity','style'):
        target=run/('source_fit_interior.png' if task=='identity' else 'target_fit_interior.png')
        panel(run/(task+'_comparison.png'), [('INPUT',run/'source_fit_interior.png'),('TARGET',target),
              ('RGB INIT -> TRAINED',run/f'rgb_{task}/final/fit_interior.png'),
              ('PERTURBED INIT -> TRAINED',run/f'perturbed_{task}/final/fit_interior.png')])
    # Same source/target pixels compared with the prior multi-source trial.
    old=run.parent/'b1-distillation-20261003/train/validation_002000/val_stairs_090.png'
    x,y,w,h=cfg['xywh'];m=cfg['margin'];a=np.array(Image.open(old))[y+m:y+h-m,x+m:x+w-m]
    Image.fromarray(a).save(run/'previous_fit_interior.png')
    target=np.array(Image.open(run/'target_fit_interior.png'))
    style_checks=[r for r in checks if r['case'].endswith('_style')]
    chosen=min(style_checks,key=lambda r:r['last_loss'])['case']
    panel(run/'old_new_style.png',[('INPUT',run/'source_fit_interior.png'),('TEACHER',run/'target_fit_interior.png'),
          ('OLD 2000 MULTI-SOURCE',run/'previous_fit_interior.png'),('NEW SINGLE-PATCH FIT',run/chosen/'final/fit_interior.png')])
    # Specific structural diagnostic: period-4 input -> spatially constant stride-4 latent.
    yy,xx=np.mgrid[:256,:256]
    tile=np.array([[40,200,80,150],[170,60,220,90],[70,180,100,230],[210,50,160,110]],np.uint8)
    pattern=np.repeat(tile[yy%4,xx%4,None],3,axis=2)
    Image.fromarray(pattern).save(run/'period4_input.png')
    periodic=[]
    for r in result['records']:
        net=load_net(r['weights'])
        with torch.inference_mode():
            outputs=net(as_tensor(pattern)[None].cuda(),return_tensors=True)
        native=outputs['rgb_head'];inner=native[:,:,80:-80,80:-80]*128+128
        spread=float((inner.amax(dim=(-2,-1))-inner.amin(dim=(-2,-1))).max())
        Image.fromarray(to_rgb(native[0])).save(run/(r['case']+'_period4_output.png'))
        periodic.append({'case':r['case'],'output_interior_channel_range_rgb255':spread,
                         'input_interior_range_rgb255':190,
                         'explanation':'Restricted period-4 diagnostic, not proof that all useful style transfer is impossible.'})
        del net
    v={'status':'NUMERICAL_CHECKS_PASS_VISUAL_REVIEW_REQUIRED','checks':checks,'selected_style_case':chosen,
       'selection_rule':'lowest final fitting objective among two style controls, not held-out selection',
       'previous_fit_teacher':score(a,target),'period4_probe':periodic,
       'all_cases_steps':sum(r['steps'] for r in checks),'not_generalization':True,'qat':False,
       'code_sha256':file_hash(Path(__file__))}
    write_json(run/'verification.json',v)
    print(json.dumps(v,indent=2),flush=True)


def render(root,run):
    v=json.loads((run/'verification.json').read_text());chosen=v['selected_style_case']
    net=load_net(run/chosen/'weights_fp32.pt')
    clips=json.loads((root/'runs/clear-style-control-20261003/video_clips.json').read_text(encoding='utf-8'))['clips']
    out=run/'videos';out.mkdir(exist_ok=False);records=[]
    for clip in clips:
        if clip['id'] not in ('stairs','plaza'):continue
        assert file_hash(clip['clip'])==clip['clip_sha256']
        name=clip['id'];path=out/(name+'_1080p.mp4');cap=cv2.VideoCapture(clip['clip'])
        pipe=encode_pipe(path,1920,1080,30);n=0;samples=[]
        while True:
            ok,bgr=cap.read()
            if not ok:break
            a=area_3to2(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
            with torch.inference_mode():b=to_rgb(net(as_tensor(a)[None].cuda())[0])
            shown=bilinear_2to3(b);pipe.stdin.write(shown.tobytes())
            if n in (0,90,210,359):
                Image.fromarray(b).save(out/f'{name}_{n:03d}_native720.png')
                Image.fromarray(shown).save(out/f'{name}_{n:03d}_preencode1080.png');samples.append(n)
            n+=1
            if n%90==0:print('RENDER',name,n,flush=True)
        cap.release();close_pipe(pipe)
        assert n==clip['validation']['frames']
        records.append({'id':name,'video':str(path),'validation':video_check(path,n),'model_calls':n,
                        'samples':samples,'role':'same_clip_as_fitting_frame' if name=='stairs' else 'untrained_source_reviewed_regression'})
        write_json(out/'report.json',{'status':'COMPLETE' if len(records)==2 else 'RENDERING','records':records,
                   'chosen':chosen,'student_postprocess':'none','pipeline':'1080 -> area720 -> B1 -> bilinear1080',
                   'source_blend':False,'temporal_blend':False,'frame_repeat':False,'model_fits_only_one_patch':True})
    print('VIDEOS_COMPLETE',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['verify','render']);args=ap.parse_args()
    root=configure_workspace();run=root/'runs'/RUN
    torch.set_num_threads(4);cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.mode=='verify':verify(run)
    else:render(root,run)


if __name__=='__main__':main()
