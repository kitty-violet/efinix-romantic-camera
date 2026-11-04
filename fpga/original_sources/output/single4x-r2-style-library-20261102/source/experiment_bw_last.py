"""Last, separate coarse-edge experiment; never counted as an accepted style."""
import json,time,copy
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import library as l
from library import D,torch,F,save_json,sha
from train import prepare_inputs,targets,cut
from render_review import rgb_batch,image_metrics,FONT

@torch.no_grad()
def initialize_edges():
    m=l.StyleModel().cuda()
    for w in m.w.values():w.zero_()
    for b in m.b.values():b.zero_()
    for name,w in m.w.items():
        if w.shape[-1]==3:w[:,0,1,1]=1
        else:
            for c in range(min(w.shape[:2])):w[c,c,0,0]=1
    m.w['stem'].zero_();m.b['stem'].fill_(64)
    for c in range(8):m.w['stem'][c,:,1,1]=torch.tensor([.1495,.2935,.057],device='cuda')
    blur=torch.tensor([[1,2,1],[2,4,2],[1,2,1]],device='cuda')/16;m.w['down_dw'][:,0]=blur
    m.w['down_pw'].zero_()
    for c in range(32):m.w['down_pw'][c,0,0,0]=1
    dx=torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]],device='cuda')/4;dy=dx.T
    kernels=[dx,-dx,dy,-dy,(dx+dy)/2,-(dx+dy)/2,(dx-dy)/2,-(dx-dy)/2]
    m.w['res1_a_dw'].zero_()
    for c,k in enumerate(kernels):m.w['res1_a_dw'][c,0]=k
    m.w['res1_b_pw'].zero_();m.w['res1_b_pw'][0,:4,0,0]=.5
    m.w['rgb'].zero_();m.w['rgb'][:,0,0,0]=-12;m.b['rgb'].fill_(127)
    m.calibrate();m.qat=True;m.leak=.002
    return m

def main():
    l.setup(20261102);folder=D/'experiments/BW_EDGE_LAST';folder.mkdir(parents=True,exist_ok=True)
    entry=next(e for e in l.read_inventory() if e['id']=='BW_LINE');teacher=l.Teacher(entry);inputs,nat=prepare_inputs();X,V=inputs;Y,_=targets(teacher,X,False);W,_=targets(teacher,V,False)
    m=initialize_edges();logs=[]
    def score():
        with torch.no_grad():
            out=torch.cat([cut(m(x)) for x in V[:nat[1]].split(4)]);target=cut(W[:nat[1]]);err=(out-target).abs()
            ink=target<0;paper=target>.8
            ie=float(err[ink].mean()*128) if ink.any() else 0;pe=float(err[paper].mean()*128) if paper.any() else 0
            return .7*ie+.3*pe,ie,pe
    initial=score();best=initial[0];state=copy.deepcopy(m.state_dict());selected=0
    for p in m.parameters():p.requires_grad_(False)
    for n in ['res1_a_dw','res1_b_pw','up1_high','rgb']:m.w[n].requires_grad_(True)
    def tangent(g):
        z=g-g[:,:,1:2,1:2];z=z.clone();z[:,:,1,1]=0;return z
    m.w['res1_a_dw'].register_hook(tangent)
    opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.0001);rng=np.random.default_rng(2026110205);start=time.perf_counter()
    for step in range(601):
        if step%50==0:
            sc,ie,pe=score();logs.append(dict(step=step,balanced_score=sc,ink_mae=ie,paper_mae=pe))
            if sc<best:best=sc;state=copy.deepcopy(m.state_dict());selected=step
            print('BW_LAST_TRY',step,round(sc,2),'ink',round(ie,2),'paper',round(pe,2),'best',selected,flush=True)
        if step==600:break
        idx=torch.as_tensor(rng.choice(nat[0],6,False),device='cuda');x=X[idx];target=cut(Y[idx]);out=cut(m(x));ink=((1-target)/2).clamp(0,1)
        opt.zero_grad(set_to_none=True);loss=((out-target).abs()*(1+5*ink)).mean()+.2*((out-target).square()*(1+5*ink)).mean();loss.backward();torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],3);opt.step()
        with torch.no_grad():
            # Tie RGB outputs. Flat fields stay white: gradient kernels have
            # zero DC; all stages before the fixed head bias have zero bias.
            m.w['rgb'].copy_(m.w['rgb'].mean(0,keepdim=True).expand_as(m.w['rgb']))
            w=m.w['res1_a_dw'];s=(m.res1_a_dw_m/131072).float()[:,None,None,None];wi=l.HalfAway.apply(w/s).clamp(-127,127);wi[:,:,1,1]=0
            centre=-wi.sum((-1,-2));assert centre.abs().max()<=127;w[:,:,1,1].copy_(centre*s[:,:,0,0])
    m.load_state_dict(state);m.leak=0;m.eval();raw=l.export(m,folder);oracle=l.independent_oracle(folder);engine=l.PackedEngine(folder)
    torch.save(dict(model=m.state_dict(),residual=m.residual,qat=True,additional_gradient_steps=600,selected_step=selected,teacher_sha256=entry['source_sha256']),folder/'selected.pt')
    scenes=json.loads((D/'review/source_manifest.json').read_text('utf-8'))['scenes'];canvas=Image.new('RGB',(1440,1200),'#111924');font=ImageFont.truetype(FONT,18);records=[]
    for row,s in enumerate(scenes):
        n=s['frames']//2;a=np.array(Image.open(l.OLD/'acceptance-four-scenes-5s-20261029/originals'/f"{s['scene']}_{n:03d}.png"));x=l.tx(a[None]);out=engine(x)
        with torch.inference_mode():assert torch.equal(out,m(x))
        b=rgb_batch(out)[0];t=rgb_batch(teacher(x))[0];gray=b.mean(2);tg=t.mean(2)
        records.append(dict(scene=s['scene'],student_black_fraction=float((gray<96).mean()),teacher_black_fraction=float((tg<96).mean()),student_white_fraction=float((gray>224).mean()),mae=image_metrics(t,b)['mae']))
        for col,(label,pic) in enumerate([('原图',a),('原教师 / 非旧B1',t),('新结构粗边缘试验 / INT8',b)]):
            ImageDraw.Draw(canvas).text((col*480+4,row*300+3),s['label']+' / '+label,font=font,fill='white');canvas.paste(Image.fromarray(pic).resize((480,270),Image.Resampling.BOX),(col*480,row*300+29))
        Image.fromarray(b).save(folder/(s['scene']+'_new.png'))
    canvas.save(folder/'教师与新结构粗边缘对照.jpg',quality=96)
    save_json(folder/'experiment.json',dict(status='EXPERIMENT_ONLY_NOT_ACCEPTED_STYLE',teacher_was_not_old_B1=True,original_fine_teacher_lines_not_claimed_reproduced=True,initial_balanced_score=initial[0],selected_balanced_score=best,selected_step=selected,actual_gradient_steps=600,records=logs,scenes=records,coefficient_sha256=sha(folder/'coefficients-padded.bin'),independent_integer_oracle=oracle,pure_exported_integer_coefficients=True,no_extra_inference_edge_filter=True,excluded_from_main_sixteen_style_package=True,seconds=time.perf_counter()-start))
    print('BW_LAST_EXPERIMENT_COMPLETE',records,flush=True)
if __name__=='__main__':main()
