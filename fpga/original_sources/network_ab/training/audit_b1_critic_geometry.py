"""Audit learned anti-blur signal on fixed TRAIN crops, without updating B1."""
import argparse,json,random
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .train_b1_trainable_luma_readout import ReadoutB1
from .train_b1_style_recovery import ConditionalCritic,appearance
from .train_b1_complete_rgb_geometry import CompleteRGBGeometryTraining
from .train_b1_contour_coherence import contour_coherence
from .whitened_b1_optimizer import transform
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--base',required=True)
    ap.add_argument('--data',required=True);ap.add_argument('--mining',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);torch.manual_seed(2026100728)
    state,net=load(a.checkpoint);_,base=load(a.base);net.__class__=ReadoutB1
    critic=ConditionalCritic(True);critic.load_state_dict(state['critic']);critic.eval().requires_grad_(False)
    rows={r['id']:r for r in m.load_data(a.data)['training']}
    mined=json.loads(Path(a.mining).read_text(encoding='utf-8'))['records'];records=[];images=[]
    dest=Path(a.out);dest.mkdir(exist_ok=False)
    for s in [mined[i] for i in (1,5,6)]:
        row=rows[s['row']['id']];x,y=s['xy'];w,h=s['wh'];sl=np.s_[y:y+h,x:x+w]
        def read(key):return torch.from_numpy(m.t.r.read(row[key])[sl].transpose(2,0,1).copy())[None].float()/128-1
        source,target=read('path'),read('target');labels=torch.from_numpy(m.load_region(row['region'])[sl].astype(np.int64))[None]
        nodes=net(source,all_nodes=True);prediction=nodes['rgb_head']/128
        with torch.no_grad():
            actual,_=m.t.r.IntegerB1(m.t.r.runtime_from_image(net.export()),device='cpu')(net.round(source*128),True)
            assert all(torch.equal(v,actual[k]) for k,v in nodes.items())
            old=base(source)[0]
        cut=lambda z:z[...,40:-40,40:-40]
        p,t,src=cut(prediction),cut(target),cut(source)
        obj=object.__new__(CompleteRGBGeometryTraining);obj.labels=labels
        structured,values=obj.additional_loss(p,t,False);curve=4*contour_coherence(p,t)
        adv=-torch.stack([q.mean() for q in critic(p,src)]).mean()
        ref=appearance(p,t,'direct')[0]
        gp=torch.autograd.grad(ref,p,retain_graph=True)[0];ga=torch.autograd.grad(adv,p,retain_graph=True)[0]
        factor=state['args']['adversarial']*(gp.norm()/ga.norm().clamp_min(1e-8)).detach().clamp(.001,1.)
        terms=dict(structured=structured,coherence=curve,adversarial=adv*factor)
        keys=('stem','res3_conv2__pw','up2_high_dw','rgb_head')
        parameters=[net.weight[k] for k in keys]+[net.bias[k] for k in keys]
        grads={k:torch.autograd.grad(v,parameters,retain_graph=True) for k,v in terms.items()}
        groups={g['node_key']:g for g in state['optimizer']['param_groups'] if 'node_key' in g}
        gradient={}
        for i,k in enumerate(('stem','res3_conv2__pw','up2_high_dw','rgb_head')):
            z={name:float(g[i].norm()) for name,g in grads.items()}
            z['adv_structured_cosine']=float((grads['structured'][i]*grads['adversarial'][i]).sum()/(grads['structured'][i].norm()*grads['adversarial'][i].norm()).clamp_min(1e-20))
            prepared={name:transform(g[i].flatten(1)-g[i+len(keys)][:,None]*groups[k]['patch_mean'],groups[k]['input_whitener'].transpose(-1,-2)) for name,g in grads.items()}
            z['prepared_norms']={name:float(g.norm()) for name,g in prepared.items()}
            z['prepared_adv_structured_cosine']=float((prepared['structured']*prepared['adversarial']).sum()/(prepared['structured'].norm()*prepared['adversarial'].norm()).clamp_min(1e-20))
            gradient[k]=z
        with torch.no_grad():
            blurred=F.avg_pool2d(F.pad(t,(2,2,2,2),mode='replicate'),5,1)
            specks=F.max_pool2d((torch.rand_like(t[:,:1])<.006).float(),3,1,1)
            artifact=(t+.18*specks*torch.where(torch.rand_like(t[:,:1])<.5,-1.,1.)).clamp(-1,127/128)
            score={name:float(torch.stack([q.mean() for q in critic(z,src)]).mean()) for name,z in dict(target=t,prediction=p,blurred=blurred,photograph=src,artifact=artifact).items()}
        records.append(dict(id=row['id'],critic_scores=score,gradient=gradient,adversarial_factor=float(factor),losses={k:float(v.detach()) for k,v in terms.items()},actual_all_nodes_error=0))
        panels=[]
        for name,z in [('source',source),('target',target),('previous',old),('critic',prediction)]:
            im=m.t.r.c.rgb_image(z[0]);im.save(dest/(row['id']+'_'+name+'.png'));panels.append(im)
        images.append(panels)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18);board=Image.new('RGB',(1280,3*350),(22,30,40));draw=ImageDraw.Draw(board)
    for j,row in enumerate(images):
        for k,im in enumerate(row):
            board.paste(im,(k*320,j*350+30));draw.text((k*320+5,j*350+3),('原片','固定目标','修改前实际B1','判别训练实际B1')[k],font=font,fill='white')
    board.save(dest/'comparison.png')
    Path(dest/'executed_audit.py').write_bytes(Path(__file__).read_bytes())
    write_json(dest/'results.json',dict(records=records,checkpoint_sha256=file_hash(a.checkpoint),base_sha256=file_hash(a.base),code_sha256=file_hash(__file__),training_crops_only=True,no_parameter_update=True,prepared_gradients='actual coordinate transformation before global clip and Adam; not the eventual nonlinear Adam update'))
    print('CRITIC_GEOMETRY_AUDIT',records,flush=True)


if __name__=='__main__':main()
