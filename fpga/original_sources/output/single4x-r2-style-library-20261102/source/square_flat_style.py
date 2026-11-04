"""Fixed flat square mosaic experiments on the unchanged S4R2 graph."""
from pathlib import Path
import copy,json,time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import library as l
from library import D,OLD,torch,F,save_json,sha
from train import prepare_inputs,cut,pixel_project
from render_review import rgb_batch,FONT

OUT=D/'experiments/square-flat-v1'

def square_flat(x,size):
    oh,ow=x.shape[-2:];ph=(-oh)%size;pw=(-ow)%size
    if ph or pw:x=F.pad(x,(0,pw,0,ph),mode='replicate')
    y=F.avg_pool2d(x,size,size)
    return y.repeat_interleave(size,-2).repeat_interleave(size,-1)[:,:,:oh,:ow]

def scenes():
    source=json.loads((D/'review/source_manifest.json').read_text('utf-8'))['scenes'];out=[]
    for s in source:
        n=s['frames']//2;p=OLD/'acceptance-four-scenes-5s-20261029/originals'/f"{s['scene']}_{n:03d}.png"
        out.append((s,np.array(Image.open(p).convert('RGB'))))
    return out

def preview():
    OUT.mkdir(parents=True,exist_ok=True);font=ImageFont.truetype(FONT,18);sizes=[4,8,16,24]
    canvas=Image.new('RGB',(2400,1200),'#101722')
    for row,(s,a) in enumerate(scenes()):
        pics=[a]+[rgb_batch(square_flat(l.tx(a[None]),n))[0] for n in sizes]
        for col,(label,pic) in enumerate(zip(['原图']+[f'{n}×{n}严格同色方格' for n in sizes],pics)):
            ImageDraw.Draw(canvas).text((col*480+4,row*300+3),s['label']+' / '+label,font=font,fill='white')
            canvas.paste(Image.fromarray(pic).resize((480,270),Image.Resampling.BOX),(col*480,row*300+29))
    canvas.save(OUT/'教师尺寸比较.jpg',quality=96)
    save_json(OUT/'teacher-design.json',dict(sizes=sizes,each_square_exact_single_rgb=True,colour='covered-pixel RGB mean',fixed_origin=True))
    print('SQUARE_TEACHER_PREVIEW_COMPLETE',flush=True)

def flat_range(out,size):
    h,w=out.shape[-2:];h=(h//size)*size;w=(w//size)*size
    z=out[:,:,:h,:w].view(out.shape[0],3,h//size,size,w//size,size)
    return float((z.amax((3,5))-z.amin((3,5))).float().mean()*128)

def evaluate(m,V,W,natural,size):
    with torch.no_grad():out=torch.cat([cut(m(x)) for x in V.split(4)]);ref=cut(W);e=(out-ref).abs().mean((1,2,3))*128
    return float(e[:natural].mean()),float(e[natural:].mean()),flat_range(out[:natural],size),flat_range(out[:natural],4)

def train(size):
    folder=OUT/f'{size}px';folder.mkdir(parents=True,exist_ok=True);l.setup(20261102);l.guard();inputs,natural=prepare_inputs();X,V=inputs
    with torch.no_grad():Y=square_flat(X,size);W=square_flat(V,size)
    m=l.StyleModel().cuda();base=torch.load(l.RECON/'single4x-r2/qat.pt',map_location='cpu',weights_only=True)
    m.load_state_dict(base['state_dict']);m.residual=copy.deepcopy(l.RESIDUAL);m.qat=False;m.leak=.002;pixel_project(m)
    rng=np.random.default_rng(2026110208+size);logs=[];start=time.perf_counter();selected=None
    for stage,steps,lr in [('fp32',600,.00018),('qat',600,.00006)]:
        m.qat=stage=='qat'
        if m.qat:m.calibrate()
        opt=torch.optim.Adam(m.parameters(),lr=lr);best=float('inf');state=copy.deepcopy(m.state_dict())
        for step in range(steps+1):
            if step%50==0:
                nat,pal,whole,micro=evaluate(m,V,W,natural[1],size);metric=nat+.2*pal+.12*whole
                logs.append(dict(stage=stage,step=step,natural_mae=nat,palette_mae=pal,target_square_internal_range=whole,micro4_internal_range=micro,metric=metric,seconds=time.perf_counter()-start))
                if metric<best:best=metric;state=copy.deepcopy(m.state_dict());selected=(stage,step)
                print('SQUARE_TRAIN',size,stage,step,'mae',round(nat,3),'range',round(whole,3),'micro4',round(micro,3),'best',selected,flush=True)
            if step==steps:break
            ids=np.r_[rng.choice(natural[0],4,False),rng.choice(np.arange(natural[0],len(X)),2,False)];idx=torch.as_tensor(ids,device='cuda')
            out=m(X[idx]);target=Y[idx];d=cut(out)-cut(target);loss=d.abs().mean()+.2*d.square().mean()
            for dim in [-1,-2]:
                go=torch.diff(cut(out),dim=dim);gt=torch.diff(cut(target),dim=dim);loss+=.3*(go.abs()*(gt.abs()<1/256)).mean()+.1*(go-gt).abs().mean()
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),3);opt.step();pixel_project(m)
        m.load_state_dict(state);pixel_project(m)
    m.qat=True;m.leak=0;m.eval();l.export(m,folder);engine=l.PackedEngine(folder);oracle=l.independent_oracle(folder)
    with torch.no_grad():
        for x in V.split(4):assert torch.equal(engine(x),m(x))
    nat,pal,whole,micro=evaluate(m,V,W,natural[1],size);torch.save(dict(model=m.state_dict(),residual=m.residual,qat=True,selected=selected),folder/'selected.pt')
    render(m,engine,size,folder)
    save_json(folder/'experiment.json',dict(status='EXPERIMENT_NOT_VISUALLY_ACCEPTED',square_size=size,teacher_square_internal_range=0,validation_natural_mae=nat,validation_palette_mae=pal,student_target_square_internal_range=whole,student_micro4_internal_range=micro,actual_gradient_steps=1200,selected=selected,coefficient_sha256=sha(folder/'coefficients-padded.bin'),independent_integer_oracle=oracle,up_high_center_only=True,no_coordinate_input=True,no_inference_filter=True,hardware_test=False,logs=logs))
    print('SQUARE_STYLE_COMPLETE',size,nat,whole,micro,flush=True)

def render(m,engine,size,folder):
    font=ImageFont.truetype(FONT,18);canvas=Image.new('RGB',(1440,1200),'#101722');records=[]
    for row,(s,a) in enumerate(scenes()):
        x=l.tx(a[None]);student=rgb_batch(engine(x))[0];teacher=rgb_batch(square_flat(x,size))[0]
        with torch.no_grad():assert torch.equal(engine(x),m(x))
        for col,(label,pic) in enumerate([('原图',a),(f'{size}×{size}严格同色教师',teacher),('S4R2整数学生',student)]):
            ImageDraw.Draw(canvas).text((col*480+4,row*300+3),s['label']+' / '+label,font=font,fill='white');canvas.paste(Image.fromarray(pic).resize((480,270),Image.Resampling.BOX),(col*480,row*300+29))
        Image.fromarray(student).save(folder/f"student_{s['scene']}.png");records.append(dict(scene=s['scene'],mae=float(np.abs(student.astype(np.float32)-teacher.astype(np.float32)).mean())))
    canvas.save(folder/'教师与学生四景对照.jpg',quality=96);save_json(folder/'static-records.json',records)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--teacher-only',action='store_true');p.add_argument('--sizes',nargs='+',type=int,default=[4,8,16]);a=p.parse_args();preview()
    if not a.teacher_only:
        for size in a.sizes:train(size)
