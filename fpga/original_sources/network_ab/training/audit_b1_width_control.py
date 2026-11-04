"""Function-preserving width control; expanded arm is NOT FPGA-compatible.

Start both arms from the same locally fitted native FP32 function. Duplicate
every hidden channel and split incoming weights with sum-preserving noise to
break symmetry. Same data, objective, optimizer reset and update budget.
Inspired by Net2WiderNet; this is a finite empirical control, not a capacity
bound or a proposed hardware replacement.
"""
import argparse,copy,json,math,shutil,time
from pathlib import Path
import numpy as np,torch
from torch import nn
from PIL import Image,ImageDraw,ImageFont
from network_ab.contract_torch import build_fp32_model
from .runtime import configure_workspace,file_hash,write_json
from .engine import set_reproducible
from .train_b1_late_region_supervision import executed_sources
from . import train_b1_minimum_area as m


@torch.no_grad()
def widen(net,factor=2):
    out=copy.deepcopy(net)
    for key,seq in net.operators.items():
        parts=[]
        for layer in seq:
            if isinstance(layer,nn.Conv2d):
                fi=1 if key=='stem' else factor;fo=1 if key=='rgb_head' else factor
                depthwise=layer.groups==layer.in_channels and layer.groups>1
                q=nn.Conv2d(layer.in_channels*fi,layer.out_channels*fo,layer.kernel_size,
                    stride=layer.stride,padding=layer.padding,groups=layer.groups*fi if depthwise else 1,
                    bias=layer.bias is not None,device=layer.weight.device)
                if depthwise:q.weight.copy_(layer.weight.repeat_interleave(fo,0))
                else:
                    w=layer.weight.repeat_interleave(fo,0).repeat_interleave(fi,1)/fi
                    if fi>1:
                        # Repeated input channels are identical at initialization.
                        # Antisymmetric outgoing perturbations preserve their sum.
                        noise=torch.randn_like(w[:,::2])*(float(layer.weight.std())*.10)
                        w[:,::2]+=noise;w[:,1::2]-=noise
                    q.weight.copy_(w)
                if layer.bias is not None:q.bias.copy_(layer.bias.repeat_interleave(fo))
                parts.append(q)
            elif isinstance(layer,nn.BatchNorm2d):
                q=nn.BatchNorm2d(layer.num_features*factor,eps=layer.eps,momentum=layer.momentum,
                    affine=layer.affine,track_running_stats=True,device=layer.weight.device)
                for k in ('weight','bias','running_mean','running_var'):
                    getattr(q,k).copy_(getattr(layer,k).repeat_interleave(factor))
                q.num_batches_tracked.copy_(layer.num_batches_tracked);parts.append(q)
            else:parts.append(copy.deepcopy(layer))
        out.operators[key]=nn.Sequential(*parts)
    out.eval();return out


def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','mining','out'):ap.add_argument('--'+k,required=True)
    ap.add_argument('--steps',type=int,default=2400)
    a=ap.parse_args();configure_workspace();set_reproducible(2026100816);torch.set_num_threads(2)
    byid={r['id']:r for r in m.load_data(a.data)['training']}
    specs=json.loads(Path(a.mining).read_text(encoding='utf-8'))['records'];specs=[specs[k] for k in (1,5,6)]
    xx=[];tt=[]
    for s in specs:
        r=byid[s['row']['id']];x,y=s['xy'];w,h=s['wh']
        for k,dst in (('path',xx),('target',tt)):
            dst.append(torch.from_numpy(m.t.r.read(r[k])[y:y+h,x:x+w].transpose(2,0,1).copy()).float()/128-1)
    x,t=torch.stack(xx).cuda(),torch.stack(tt).cuda();cut=lambda z:z[...,40:-40,40:-40]
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    base=build_fp32_model().cuda().eval();base.load_state_dict(state['model'])
    wide=widen(base)
    with torch.no_grad():
        a0=base(x,return_tensors=True);a1=wide(x,return_tensors=True);errors={}
        for k,z in a0.items():
            ref=z if k in ('input','rgb_head') else z.repeat_interleave(2,1)
            errors[k]=float((ref-a1[k]).abs().max())
        err=errors['rgb_head']*128;assert err<.02 and max(errors.values())<.01,errors
    out=Path(a.out);out.mkdir(exist_ok=False)
    paths=sorted(set(executed_sources(type(base))[0]+[Path(__file__)]));hashes={str(p):file_hash(p) for p in paths}
    for p in paths:shutil.copy2(p,out/p.name)
    write_json(out/'config.json',dict(args=vars(a),specs=specs,code=hashes,checkpoint_sha256=file_hash(a.checkpoint),
        data_sha256=file_hash(a.data),initial_output_error_byte=err,initial_node_errors=errors,
        original_training_params=sum(p.numel() for p in base.parameters()),
        widened_training_params=sum(p.numel() for p in wide.parameters()),
        expanded_is_fpga_compatible=False,hardware_changed=False,only_three_train_crops=True,
        fixed_bn_statistics=True,initial_function_preserved_on_test_batch=True))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13);histories={};outputs={};started=time.monotonic()
    for name,net in (('width1',base),('width2',wide)):
        net.eval();initial=copy.deepcopy(net.state_dict());opt=torch.optim.Adam(net.parameters(),lr=.0003)
        def update(step):
            opt.zero_grad();p=net(x);loss=(cut(p)-cut(t)).abs().mean();loss.backward()
            gn=torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);assert gn>0
            for g in opt.param_groups:g['lr']=.0001+.0002*.5*(1+math.cos(math.pi*step/a.steps))
            opt.step();return float(loss.detach()),float(gn)
        update(0);torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict()),out/(name+'_restore.pt'))
        expected=update(1);expected_state=copy.deepcopy(net.state_dict())
        disk=torch.load(out/(name+'_restore.pt'),map_location='cuda',weights_only=True)
        net.load_state_dict(disk['model']);opt.load_state_dict(disk['optimizer']);actual=update(1)
        re=max(float((v-net.state_dict()[k]).abs().max()) for k,v in expected_state.items());assert re==0 and expected==actual
        net.load_state_dict(initial);opt=torch.optim.Adam(net.parameters(),lr=.0003);history=[]
        write_json(out/(name+'_preflight.json'),dict(status='PASS',gpu_minibatch_backward=True,disk_next_step_error=re))
        print('WIDTH_CONTROL_PREFLIGHT',name,'PASS disk0',flush=True)
        for step in range(a.steps+1):
            if step%600==0 or step==a.steps:
                assert all(file_hash(p)==v for p,v in hashes.items())
                with torch.no_grad():p=net(x);mae=float((cut(p)-cut(t)).abs().mean()*128)
                history.append(dict(step=step,mae=mae));write_json(out/(name+'_history.json'),history)
                torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),step=step,width=1 if name=='width1' else 2,
                    local_fit_only=True,quantization_tested=False,fpga_compatible=False,visual_acceptance=False),out/f'{name}_{step:06d}.pt')
                board=Image.new('RGB',(480,792),'#17202a');draw=ImageDraw.Draw(board)
                for i in range(3):
                    for j,(label,z) in enumerate((('目标 中心240',t),(name+' FP32诊断',p))):
                        board.paste(m.t.r.c.rgb_image(cut(z)[i]),(240*j,264*i+24));draw.text((240*j+3,264*i+3),label,font=font,fill='white')
                board.save(out/f'{name}_{step:06d}.png');print('WIDTH_CONTROL',name,step,mae,flush=True)
            if step==a.steps:break
            update(step)
            assert all(torch.equal(v,initial[k]) for k,v in net.named_buffers())
        histories[name]=history;outputs[name]=p.detach().cpu();del disk,expected_state,opt
        net.cpu();torch.cuda.empty_cache()
    board=Image.new('RGB',(720,792),'#17202a');draw=ImageDraw.Draw(board)
    for i in range(3):
        for j,(name,z) in enumerate((('固定目标',t.cpu()),('原宽度 FP32诊断',outputs['width1']),('双宽度 FP32·不能上板',outputs['width2']))):
            board.paste(m.t.r.c.rgb_image(cut(z)[i]),(240*j,264*i+24));draw.text((240*j+3,264*i+3),name,font=font,fill='white')
    board.save(out/'comparison.png');write_json(out/'results.json',dict(histories=histories,seconds=time.monotonic()-started,
        expanded_is_fpga_compatible=False,capacity_limit_proven=False,visual_acceptance=False))


if __name__=='__main__':main()
