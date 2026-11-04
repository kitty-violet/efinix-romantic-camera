"""Learn only visible region colours with alternating two-branch assignment.

Old supervision required a particular hidden neighbour colour even where
that neighbour is not used in the output. This controlled run replaces that
latent assignment with the closer predicted RGB branch and its common gate.
The fixed byte-exact two-colour renderer is preserved; no new inference op.
"""
import argparse,copy,json,shutil,time
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .train_b1_two_colour_fields import TwoColourB1
from .train_b1_trainable_luma_readout import ReadoutB1
from .whitened_b1_optimizer import WhitenedB1Adam,patch_statistics
from .runtime import configure_workspace,write_json,file_hash,contract_guard
from .engine import set_reproducible
from .train_b1_late_region_supervision import executed_sources
from . import train_b1_minimum_area as m
from .probe_b1_two_colour_geometry import latent_full


class RoutedColourB1(TwoColourB1,ReadoutB1):
    pass


def mixture_loss(fields,target):
    a=fields[:,:3]/64-1;b=fields[:,3:6]/64-1
    ca=(a-target).abs().mean(1,keepdim=True);cb=(b-target).abs().mean(1,keepdim=True)
    # Stable ties follow the current gate, not an arbitrary always-A bias.
    choose=((cb<ca)|((cb==ca)&(fields[:,6:7]>=64))).detach().float()
    colour=((1-choose)*ca+choose*cb).mean()
    certainty=((ca-cb).abs()/.1).clamp(0,1).detach()
    gate=F.binary_cross_entropy_with_logits((fields[:,6:7]-63.5)/4,choose,reduction='none')
    gate=(gate*certainty).sum()/certainty.sum().clamp_min(1)
    return colour,gate


def regression():
    z=torch.zeros(1,7,4,4);z[:,:3]=32;z[:,3:6]=96;z[:,6]=60
    t=torch.full((1,3,4,4),-.5);z.requires_grad_();c,g=mixture_loss(z,t)
    assert c==0 and g>0;grad=torch.autograd.grad(g,z)[0];assert grad[:,6].mean()>0 and grad[:,:6].abs().sum()==0
    swap=z.detach().clone();swap[:,:3]=z.detach()[:,3:6];swap[:,3:6]=z.detach()[:,:3];swap[:,6]=127-z.detach()[:,6]
    cc,gg=mixture_loss(swap,t);assert c==cc and torch.allclose(g,gg)
    return dict(visible_colour_exact_zero=True,common_gate_gradient_correct=True,invisible_colour_not_forced=True,branch_swap_invariant=True)


def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','mining','out'):ap.add_argument('--'+k,required=True)
    ap.add_argument('--steps',type=int,default=2400)
    ap.add_argument('--pair-set-weight',type=float,default=0.)
    ap.add_argument('--field-smooth-weight',type=float,default=0.)
    a=ap.parse_args()
    configure_workspace();set_reproducible(81028);torch.set_num_threads(2);checks=regression()
    out=Path(a.out);out.mkdir(exist_ok=False);state,net=load(a.checkpoint);net.cuda();net.__class__=RoutedColourB1
    assert state['two_colour_fields'] and state['coverage_layout']=='two_rgb_sides_and_signed_contour'
    rows={r['id']:r for r in m.load_data(a.data)['training']};ss=json.loads(Path(a.mining).read_text('utf-8'))['records'];specs=[ss[i] for i in (1,5,6)]
    xs=[];ts=[];pairs=[]
    for s in specs:
        row=rows[s['row']['id']];x,y=s['xy'];w,h=s['wh']
        for k,d in (('path',xs),('target',ts)):d.append(m.t.r.read(row[k])[y:y+h,x:x+w])
        if a.pair_set_weight:
            assert file_hash(row['hard_region_target'])==row['native_luma_colour_source_sha256']
            pairs.append(latent_full(row['hard_region_target'],row['region'])[y:y+h,x:x+w,:6])
    tensor=lambda v:torch.from_numpy(np.stack(v).transpose(0,3,1,2).copy()).float().cuda()/128-1
    x=tensor(xs);t=tensor(ts);cut=lambda z:z[...,40:-40,40:-40]
    if a.pair_set_weight:
        pair_truth=torch.from_numpy(np.stack(pairs).transpose(0,3,1,2).copy()).float().cuda()/64-1
        pair_truth=F.avg_pool2d(cut(pair_truth),2)
        pair_mask=((pair_truth[:,:3]-pair_truth[:,3:6]).abs().mean(1,keepdim=True)>.06).float()
    with torch.no_grad():old=net(x)[0]
    fixed=False;groups=[];frozen={}
    for k,w in net.weight.items():
        if k=='up1_high_dw':fixed=True
        w.requires_grad_(not fixed);net.bias[k].requires_grad_(not fixed)
        if fixed:frozen[k]=(w.detach().clone(),net.bias[k].detach().clone())
        else:
            s=(getattr(net,k+'_m')/131072).float().flatten()
            groups.append(dict(params=[w],lr=.0001,row_gain=(s*256)[:,None,None,None]))
            groups.append(dict(params=[net.bias[k]],lr=.01))
    with torch.no_grad():stats=patch_statistics(net,x)
    opt=WhitenedB1Adam(groups,net,stats)
    def update():
        opt.zero_grad(set_to_none=True);nodes=net(x,all_nodes=True);p=cut(nodes['rgb_head']/128)
        fields=nodes['up1_high_dw'][:,:7,20:-20,20:-20];want=F.avg_pool2d(cut(t),2)
        colour,gate=mixture_loss(fields,want);rgb=(p-cut(t)).abs().mean()
        smooth=sum(torch.diff(fields[:,:6]/64,dim=d).abs().mean() for d in (-1,-2))/2
        pair=rgb*0
        if a.pair_set_weight:
            ca=fields[:,:3]/64-1;cb=fields[:,3:6]/64-1;ta=pair_truth[:,:3];tb=pair_truth[:,3:6]
            direct=(ca-ta).abs().mean(1,keepdim=True)+(cb-tb).abs().mean(1,keepdim=True)
            swapped=(ca-tb).abs().mean(1,keepdim=True)+(cb-ta).abs().mean(1,keepdim=True)
            pick=(direct<=swapped).detach()
            pair=(torch.where(pick,direct,swapped)*pair_mask).sum()/pair_mask.sum().clamp_min(1)
        loss=4*colour+.2*gate+2*rgb+a.pair_set_weight*pair+a.field_smooth_weight*smooth;loss.backward();opt.prepare_gradients()
        gn=torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);assert gn>0 and torch.isfinite(loss);opt.step()
        return tuple(float(v.detach()) for v in (loss,colour,gate,rgb,pair,smooth))
    initial=copy.deepcopy(net.state_dict());io=copy.deepcopy(opt.state_dict())
    update();torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict()),out/'restore_probe.pt');expected=update();snap=copy.deepcopy(net.state_dict())
    disk=torch.load(out/'restore_probe.pt',map_location='cuda',weights_only=True);net.load_state_dict(disk['model']);opt.load_state_dict(disk['optimizer']);observed=update()
    err=max(float((v-net.state_dict()[k]).abs().max()) for k,v in snap.items());assert err==0 and expected==observed
    net.load_state_dict(initial);opt.load_state_dict(io)
    paths=sorted(set(executed_sources(type(net))[0]+[Path(__file__)]));hashes={str(p):file_hash(p) for p in paths}
    for p in paths:shutil.copy2(p,out/p.name)
    write_json(out/'preflight.json',dict(regression=checks,gpu_forward_backward_minibatch=True,disk_next_step_error=err))
    write_json(out/'config.json',dict(args=vars(a),code=hashes,specs=specs,local_TRAIN_only=True,extra_inference_ops=0,hidden_neighbour_field_supervision=False))
    print('TWO_COLOUR_EM_PREFLIGHT realGPU/backward/minibatch/disk0',flush=True)
    history=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12);start=time.monotonic()
    def review(step):
        for k,(w,b) in frozen.items():assert torch.equal(w,net.weight[k]) and torch.equal(b,net.bias[k])
        with torch.no_grad():
            raw=net.export();assert len(raw)==20480;nodes=net(x,all_nodes=True)
            actual,_=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cuda')(net.round(x*128),True)
            assert all(torch.equal(v,actual[k]) for k,v in nodes.items());p=actual['rgb_head']/128
        board=Image.new('RGB',(960,792),'#17202a');draw=ImageDraw.Draw(board)
        for i in range(3):
            for j,(title,z) in enumerate((('原片',x),('固定目标',t),('旧双侧颜色B1',old),('可见颜色训练·实际B1',p))):
                board.paste(m.t.r.c.rgb_image(cut(z)[i]),(j*240,i*264+24));draw.text((j*240+3,i*264+3),title,font=font,fill='white')
        board.save(out/f'comparison_{step:06d}.png');(out/f'coefficients_{step:06d}.bin').write_bytes(raw)
        torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),step=step,contract_sha256=state.get('contract_sha256',m.t.r.CONTRACT_SHA),
            residual_encoding=net.package['residual_encoding'],qat=True,two_colour_fields=True,coverage_layout='two_rgb_sides_and_signed_contour',
            local_fit_only=True,visual_acceptance=False,visible_colour_assignment=True),out/f'candidate_{step:06d}.pt')
        rec=dict(step=step,mae=float(cut(p-t).abs().mean()*128),all_nodes_error=0);history.append(rec);write_json(out/'history.json',history);print('TWO_COLOUR_EM_REVIEW',rec,flush=True)
    review(0)
    for step in range(a.steps):
        v=update()
        if (step+1)%100==0:print('TWO_COLOUR_EM_STEP',step+1,v,flush=True)
        if (step+1)%400==0 or step+1==a.steps:
            assert all(file_hash(k)==v for k,v in hashes.items());review(step+1)
            if (out/'STOP_REQUEST').exists():break
    write_json(out/'completion.json',dict(history=history,seconds=time.monotonic()-start,local_TRAIN_only=True,visual_acceptance=False,video_not_validated=True))

if __name__=='__main__':main()
