"""Independent fixed triangular flat-fill experiment for S4R2.

The triangle teacher is used only while fitting and for evaluation. Exported
student inference remains the existing fifteen convolutions, two residuals and
nearest x4. No coordinate channel, mask, LUT or post filter is added.
"""
from pathlib import Path
import copy,json,time,hashlib
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import library as l
from library import D,OLD,torch,F,save_json,sha
from train import prepare_inputs,cut
from render_review import rgb_batch,FONT

OUT=D/'experiments/triangle-flat-v1'

def triangle_flat(x,size=16):
    """Fill alternating right triangles with their exact mean RGB colour."""
    oh,ow=x.shape[-2:];ph=(-oh)%size;pw=(-ow)%size
    if ph or pw:x=F.pad(x,(0,pw,0,ph),mode='replicate')
    b,c,h,w=x.shape
    th,tw=h//size,w//size
    z=x.view(b,c,th,size,tw,size).permute(0,1,2,4,3,5)
    yy,xx=torch.meshgrid(torch.arange(size,device=x.device),torch.arange(size,device=x.device),indexing='ij')
    parity=(torch.arange(th,device=x.device)[:,None]+torch.arange(tw,device=x.device)[None,:])&1
    base=(yy[:,None,:]>=xx[None,:,:]).permute(1,0,2) if False else (yy>=xx)
    mask=torch.where(parity[:,:,None,None].bool(),base.flip(-1)[None,None],base[None,None])
    mask=mask[None,None].to(x.dtype)
    count=mask.sum((-1,-2),keepdim=True)
    a=(z*mask).sum((-1,-2),keepdim=True)/count
    bmean=(z*(1-mask)).sum((-1,-2),keepdim=True)/(size*size-count)
    y=a*mask+bmean*(1-mask)
    return y.permute(0,1,2,4,3,5).reshape(b,c,h,w)[:,:,:oh,:ow]

def load_mid_scenes():
    sources=json.loads((D/'review/source_manifest.json').read_text('utf-8'))['scenes'];rows=[]
    for s in sources:
        n=s['frames']//2
        p=OLD/'acceptance-four-scenes-5s-20261029/originals'/f"{s['scene']}_{n:03d}.png"
        rows.append((s,np.array(Image.open(p).convert('RGB'))))
    return rows

def teacher_preview():
    OUT.mkdir(parents=True,exist_ok=True);font=ImageFont.truetype(FONT,18)
    rows=load_mid_scenes();sizes=[8,16,24]
    canvas=Image.new('RGB',(1920,1200),'#101722')
    records=[]
    for r,(s,a) in enumerate(rows):
        pics=[a]
        for size in sizes:
            x=l.tx(a[None]);y=rgb_batch(triangle_flat(x,size))[0];pics.append(y)
            if size==16:Image.fromarray(y).save(OUT/f"teacher_{s['scene']}.png")
        for col,(label,pic) in enumerate(zip(['原图','8px三角','16px三角','24px三角'],pics)):
            ImageDraw.Draw(canvas).text((col*480+4,r*300+3),s['label']+' / '+label,font=font,fill='white')
            canvas.paste(Image.fromarray(pic).resize((480,270),Image.Resampling.BOX),(col*480,r*300+29))
        records.append(dict(scene=s['scene'],selected_size=16))
    canvas.save(OUT/'教师尺寸比较.jpg',quality=96)
    save_json(OUT/'teacher-design.json',dict(status='TEACHER_PREVIEW',selected_triangle_square_size=16,triangles_per_square=2,alternating_diagonal=True,each_triangle_exact_single_rgb=True,records=records,inference_filter=False))
    print('TRIANGLE_TEACHER_PREVIEW_COMPLETE',OUT/'教师尺寸比较.jpg',flush=True)

def score_model(m,V,target,natural_count,size):
    with torch.no_grad():
        out=torch.cat([cut(m(x)) for x in V.split(4)]);ref=cut(target)
        mae=(out-ref).abs().mean((1,2,3))*128
        # Exact triangle flatness: mean maximum RGB range within each target
        # triangle. Zero is ideal. This is measured, not used at inference.
        flat=[]
        for y in out[:natural_count]:
            z=y[None];s=size;h,w=z.shape[-2:];h=(h//s)*s;w=(w//s)*s;z=z[:,:,:h,:w];th,tw=h//s,w//s
            q=z.view(1,3,th,s,tw,s).permute(0,1,2,4,3,5)
            yy,xx=torch.meshgrid(torch.arange(s,device=z.device),torch.arange(s,device=z.device),indexing='ij')
            parity=(torch.arange(th,device=z.device)[:,None]+torch.arange(tw,device=z.device)[None,:])&1
            base=yy>=xx;mask=torch.where(parity[:,:,None,None].bool(),base.flip(-1)[None,None],base[None,None])
            vals=[]
            for mk in [mask,~mask]:
                expanded=mk[None,None].expand_as(q)
                vmax=q.masked_fill(~expanded,-1e9).amax((-1,-2))
                vmin=q.masked_fill(~expanded,1e9).amin((-1,-2))
                vals.append((vmax-vmin).mean())
            flat.append(torch.stack(vals).mean()*128)
        return float(mae[:natural_count].mean()),float(mae[natural_count:].mean()),float(torch.stack(flat).mean())

def train(size):
    l.setup(20261102);l.guard();folder=OUT/f'{size}px';folder.mkdir(parents=True,exist_ok=True)
    inputs,natural=prepare_inputs();X,V=inputs
    with torch.no_grad():Y=triangle_flat(X,size);W=triangle_flat(V,size)
    m=l.StyleModel().cuda();base=torch.load(l.RECON/'single4x-r2/qat.pt',map_location='cpu',weights_only=True)
    m.load_state_dict(base['state_dict']);m.residual=copy.deepcopy(l.RESIDUAL);m.qat=False;m.leak=.002
    rng=np.random.default_rng(2026110207);logs=[];start=time.perf_counter();best=float('inf');state=None;selected=None
    stages=[('fp32',700,.00018),('qat',700,.00006)]
    for stage,steps,lr in stages:
        m.qat=stage=='qat'
        if m.qat:m.calibrate()
        opt=torch.optim.Adam(m.parameters(),lr=lr);stage_best=float('inf');stage_state=copy.deepcopy(m.state_dict())
        for step in range(steps+1):
            if step%50==0:
                nat,pal,flat=score_model(m,V,W,natural[1],size);metric=nat+.2*pal+.15*flat
                logs.append(dict(stage=stage,step=step,natural_mae=nat,palette_mae=pal,triangle_internal_range=flat,metric=metric,seconds=time.perf_counter()-start))
                if metric<stage_best:stage_best=metric;stage_state=copy.deepcopy(m.state_dict());selected=(stage,step)
                print('TRIANGLE_TRAIN',size,stage,step,'mae',round(nat,3),'flat',round(flat,3),'best',selected,flush=True)
            if step==steps:break
            ids=np.r_[rng.choice(natural[0],4,replace=False),rng.choice(np.arange(natural[0],len(X)),2,replace=False)]
            idx=torch.as_tensor(ids,device='cuda');x=X[idx];target=Y[idx]
            opt.zero_grad(set_to_none=True);out=m(x);d=cut(out)-cut(target)
            loss=d.abs().mean()+.2*d.square().mean()
            # Penalize output gradients where the teacher triangle is flat;
            # keep the actual triangle boundary gradients.
            for dim in [-1,-2]:
                go=torch.diff(cut(out),dim=dim);gt=torch.diff(cut(target),dim=dim);mask=(gt.abs()<1/256).float()
                loss+=.25*(go.abs()*mask).mean()+.15*(go-gt).abs().mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),3);opt.step()
        m.load_state_dict(stage_state)
    m.qat=True;m.leak=0;m.eval();l.export(m,folder);engine=l.PackedEngine(folder);oracle=l.independent_oracle(folder)
    with torch.no_grad():
        for x in V.split(4):assert torch.equal(engine(x),m(x))
    nat,pal,flat=score_model(m,V,W,natural[1],size);torch.save(dict(model=m.state_dict(),residual=m.residual,qat=True,selected=selected),folder/'selected.pt')
    render_static(m,engine,size,folder)
    save_json(folder/'experiment.json',dict(status='EXPERIMENT_NOT_VISUALLY_ACCEPTED',triangle_size=size,each_teacher_triangle_exact_single_rgb=True,actual_gradient_steps=1400,selected=selected,validation_natural_mae=nat,validation_palette_mae=pal,student_triangle_internal_range=flat,logs=logs,coefficient_sha256=sha(folder/'coefficients-padded.bin'),independent_integer_oracle=oracle,no_coordinate_input=True,no_inference_filter=True,hardware_test=False))
    print('TRIANGLE_STYLE_EXPERIMENT_COMPLETE',size,'mae',nat,'flat',flat,flush=True)

def render_static(m,engine,size,folder):
    rows=load_mid_scenes();font=ImageFont.truetype(FONT,18);canvas=Image.new('RGB',(1440,1200),'#101722');recs=[]
    for r,(s,a) in enumerate(rows):
        x=l.tx(a[None]);student=rgb_batch(engine(x))[0];teacher=rgb_batch(triangle_flat(x,size))[0]
        with torch.no_grad():assert torch.equal(engine(x),m(x))
        for col,(label,pic) in enumerate([('原图',a),(f'固定{size}px三角教师',teacher),('S4R2整数学生',student)]):
            ImageDraw.Draw(canvas).text((col*480+4,r*300+3),s['label']+' / '+label,font=font,fill='white')
            canvas.paste(Image.fromarray(pic).resize((480,270),Image.Resampling.BOX),(col*480,r*300+29))
        Image.fromarray(student).save(folder/f"student_{s['scene']}.png")
        recs.append(dict(scene=s['scene'],teacher_student_mae=float(np.abs(teacher.astype(np.float32)-student.astype(np.float32)).mean())))
    canvas.save(folder/'教师与学生四景对照.jpg',quality=96);save_json(folder/'static-records.json',recs)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--teacher-only',action='store_true');p.add_argument('--sizes',nargs='+',type=int,default=[16,24]);a=p.parse_args()
    teacher_preview()
    if not a.teacher_only:
        for size in a.sizes:train(size)
