"""Fixed S4R2 square experiments using max/centre/min target colours."""
import copy,time
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import library as l
from library import D,torch,F,save_json,sha
from train import prepare_inputs,cut,pixel_project
from square_flat_style import scenes
from render_review import rgb_batch,FONT

def square_reduce(x,size,mode):
    oh,ow=x.shape[-2:];ph=(-oh)%size;pw=(-ow)%size
    if ph or pw:x=F.pad(x,(0,pw,0,ph),mode='replicate')
    if mode=='max':y=F.max_pool2d(x,size,size)
    elif mode=='min':y=-F.max_pool2d(-x,size,size)
    elif mode=='center':
        z=x.unfold(2,size,size).unfold(3,size,size)
        # Even-sized cells have four geometric centre pixels. Use one fixed
        # sample: the upper-left centre at floor((size-1)/2) in both axes.
        centre=(size-1)//2;y=z[...,centre,centre]
    else:raise ValueError(mode)
    return y.repeat_interleave(size,-2).repeat_interleave(size,-1)[:,:,:oh,:ow]

def evaluate(m,V,W,natural,size):
    with torch.no_grad():out=torch.cat([cut(m(x)) for x in V.split(4)]);ref=cut(W);e=(out-ref).abs().mean((1,2,3))*128
    h,w=out.shape[-2:];h=(h//size)*size;w=(w//size)*size
    z=out[:natural,:,:h,:w].view(natural,3,h//size,size,w//size,size)
    whole=float((z.amax((3,5))-z.amin((3,5))).float().mean()*128)
    z4=out[:natural].view(natural,3,out.shape[-2]//4,4,out.shape[-1]//4,4)
    micro=float((z4.amax((3,5))-z4.amin((3,5))).float().mean()*128)
    return float(e[:natural].mean()),float(e[natural:].mean()),whole,micro

def train(size,mode):
    l.setup(20261102);l.guard();root=D/'experiments'/f'square-{mode}-v1';folder=root/f'{size}px';folder.mkdir(parents=True,exist_ok=True);inputs,natural=prepare_inputs();X,V=inputs
    with torch.no_grad():Y=square_reduce(X,size,mode);W=square_reduce(V,size,mode)
    m=l.StyleModel().cuda();base=torch.load(D/'experiments/square-flat-v1/4px/selected.pt',map_location='cpu',weights_only=True)
    m.load_state_dict(base['model']);m.residual=copy.deepcopy(l.RESIDUAL);m.qat=False;m.leak=.002;pixel_project(m)
    rng=np.random.default_rng(2026110212+size);logs=[];start=time.perf_counter();selected=None
    for stage,steps,lr in [('fp32',600,.00016),('qat',600,.000055)]:
        m.qat=stage=='qat'
        if m.qat:m.calibrate()
        opt=torch.optim.Adam(m.parameters(),lr=lr);best=float('inf');state=copy.deepcopy(m.state_dict())
        for step in range(steps+1):
            if step%50==0:
                nat,pal,whole,micro=evaluate(m,V,W,natural[1],size);metric=nat+.2*pal+.15*whole
                logs.append(dict(stage=stage,step=step,natural_mae=nat,palette_mae=pal,target_square_internal_range=whole,micro4_internal_range=micro,metric=metric,seconds=time.perf_counter()-start))
                if metric<best:best=metric;state=copy.deepcopy(m.state_dict());selected=(stage,step)
                print('SQUARE_REDUCE_TRAIN',mode,size,stage,step,'mae',round(nat,3),'range',round(whole,3),'micro4',round(micro,3),'best',selected,flush=True)
            if step==steps:break
            ids=np.r_[rng.choice(natural[0],4,False),rng.choice(np.arange(natural[0],len(X)),2,False)];idx=torch.as_tensor(ids,device='cuda')
            out=m(X[idx]);target=Y[idx];d=cut(out)-cut(target);loss=d.abs().mean()+.2*d.square().mean()
            for dim in [-1,-2]:
                go=torch.diff(cut(out),dim=dim);gt=torch.diff(cut(target),dim=dim);loss+=.25*(go.abs()*(gt.abs()<1/256)).mean()+.1*(go-gt).abs().mean()
            opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),3);opt.step();pixel_project(m)
        m.load_state_dict(state);pixel_project(m)
    m.qat=True;m.leak=0;m.eval();l.export(m,folder);engine=l.PackedEngine(folder);oracle=l.independent_oracle(folder)
    with torch.no_grad():
        for x in V.split(4):assert torch.equal(engine(x),m(x))
    nat,pal,whole,micro=evaluate(m,V,W,natural[1],size);torch.save(dict(model=m.state_dict(),residual=m.residual,qat=True,selected=selected),folder/'selected.pt')
    render(m,engine,size,folder,mode)
    save_json(folder/'experiment.json',dict(status='EXPERIMENT_NOT_VISUALLY_ACCEPTED',square_size=size,teacher_reduction='per_channel_'+mode,teacher_square_internal_range=0,validation_natural_mae=nat,validation_palette_mae=pal,student_target_square_internal_range=whole,student_micro4_internal_range=micro,actual_gradient_steps=1200,selected=selected,coefficient_sha256=sha(folder/'coefficients-padded.bin'),independent_integer_oracle=oracle,fixed_s4r2_operators=True,no_inference_filter=True,hardware_test=False,logs=logs))
    print('SQUARE_REDUCE_COMPLETE',mode,size,nat,whole,micro,flush=True)

def render(m,engine,size,folder,mode):
    font=ImageFont.truetype(FONT,18);canvas=Image.new('RGB',(1440,1200),'#101722');records=[]
    for row,(s,a) in enumerate(scenes()):
        x=l.tx(a[None]);student=rgb_batch(engine(x))[0];teacher=rgb_batch(square_reduce(x,size,mode))[0]
        with torch.no_grad():assert torch.equal(engine(x),m(x))
        cn={'max':'最大值','center':'中心像素','min':'最小值'}[mode]
        for col,(label,pic) in enumerate([('原图',a),(f'{size}×{size}通道{cn}教师',teacher),('固定S4R2整数学生',student)]):
            ImageDraw.Draw(canvas).text((col*480+4,row*300+3),s['label']+' / '+label,font=font,fill='white');canvas.paste(Image.fromarray(pic).resize((480,270),Image.Resampling.BOX),(col*480,row*300+29))
        Image.fromarray(student).save(folder/f'student_{s["scene"]}.png');records.append(dict(scene=s['scene'],mae=float(np.abs(student.astype(np.float32)-teacher.astype(np.float32)).mean())))
    canvas.save(folder/'教师与学生四景对照.jpg',quality=96);save_json(folder/'static-records.json',records)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--sizes',nargs='+',type=int,default=[4,8,16]);p.add_argument('--mode',choices=['max','center','min'],default='max');a=p.parse_args()
    for size in a.sizes:train(size,a.mode)
