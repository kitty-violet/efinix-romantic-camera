"""Local FP32 diagnostic of the original B1 training graph from a fresh start.

No inherited ordinal decoder, fixed scales, STE or custom optimizer. Uses the
contract's existing Conv/BN/ReLU graph and ordinary Adam. BN is training-only
and must be folded/quantized separately before any deployability claim.
Three TRAIN crops, not validation or an estimate of a global capacity bound.
"""
import argparse,copy,json,math,shutil,time
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from network_ab.contract_torch import build_fp32_model
from .train_b1_coherent_patch_native import objective
from .train_b1_contour_coherence import contour_coherence
from .runtime import write_json,file_hash,configure_workspace
from .engine import set_reproducible
from .train_b1_late_region_supervision import executed_sources
from . import train_b1_minimum_area as m


def sparse_pixel_loss(p,t):
    """Reuse the earlier sparse-gradient principle on this plain FP32 control.

    This is not a new solver or an inference filter. RGB shares one gradient
    magnitude, so independently displaced colour edges pay separately.
    """
    cost=[]
    for axis in (-1,-2):
        g=torch.diff(p,dim=axis).square().mean(1)
        cost.append((g/(g+(4/128)**2)).mean())
    return (p-t).abs().mean()+.04*torch.stack(cost).mean()


def sparse_regression():
    from torch.nn import functional as F
    t=torch.zeros(1,3,64,96);t[...,48:]=32/128
    blur=F.avg_pool2d(F.pad(t,(4,4,4,4),mode='replicate'),9,1)
    assert sparse_pixel_loss(t,t)<sparse_pixel_loss(blur,t)
    for width in (1,2,3,4):
        q=torch.zeros_like(t);q[...,48:48+width]=24/128
        assert sparse_pixel_loss(q,q)<sparse_pixel_loss(q*0,q)
    p=blur.clone().requires_grad_();sparse_pixel_loss(p,t).backward()
    assert torch.isfinite(p.grad).all() and (p.grad*(t-p.detach())).sum()<0
    assert sparse_pixel_loss(t*0,t*0)==0
    print('PLAIN_SPARSE_CHECK constant/blur/thin1to4/finite_gradient PASS',flush=True)


def main():
    ap=argparse.ArgumentParser()
    for k in ('data','mining','out'):ap.add_argument('--'+k,required=True)
    ap.add_argument('--steps',type=int,default=2000);ap.add_argument('--interval',type=int,default=400)
    ap.add_argument('--device',choices=('cpu','cuda'),default='cpu');ap.add_argument('--resume',default='')
    ap.add_argument('--reference',default='',help='Optional bounded-source320px reference directory; local diagnostic only')
    ap.add_argument('--reset-optimizer',action='store_true',help='Reuse native graph weights/buffers but reset Adam for changed targets')
    ap.add_argument('--loss',choices=('coherent','pixel_l1','pixel_mse','pixel_sparse'),default='coherent')
    a=ap.parse_args();configure_workspace();set_reproducible(2026100811);torch.set_num_threads(2)
    if a.loss=='pixel_sparse':sparse_regression()
    rows={r['id']:r for r in m.load_data(a.data)['training']}
    mining=json.loads(Path(a.mining).read_text(encoding='utf-8'))['records'];specs=[mining[k] for k in (1,5,6)]
    xx=[];tt=[];ll=[];reference_hashes={}
    for spec in specs:
        row=rows[spec['row']['id']];x,y=spec['xy'];w,h=spec['wh']
        for key,dest in (('path',xx),('target',tt)):
            dest.append(torch.from_numpy(m.t.r.read(row[key])[y:y+h,x:x+w].transpose(2,0,1).copy()).float()/128-1)
        ll.append(torch.from_numpy(m.load_region(row['region'])[y:y+h,x:x+w].astype(np.int64)))
        if a.reference:
            import cv2
            path=Path(a.reference)/(row['id']+'_3.png');rgb=np.array(Image.open(path).convert('RGB'))
            assert rgb.shape==(h,w,3)
            tt[-1]=torch.from_numpy(rgb.transpose(2,0,1).copy()).float()/128-1
            # Evaluation seam labels only; pixel loss never consumes labels.
            lab=cv2.cvtColor(rgb.astype(np.float32)/255,cv2.COLOR_RGB2Lab)
            bins=np.floor(lab/np.array([8.,6.,6.])+.5).astype(np.int32)
            _,codes=np.unique(bins.reshape(-1,3),axis=0,return_inverse=True)
            ll[-1]=torch.from_numpy(codes.reshape(h,w).astype(np.int64))
            reference_hashes[str(path)]=file_hash(path)
    x,t,lab=[v.to(a.device) for v in (torch.stack(xx),torch.stack(tt),torch.stack(ll))];cut=lambda z:z[...,40:-40,40:-40]
    net=build_fp32_model().to(a.device)
    opt=torch.optim.Adam(net.parameters(),lr=.001)
    start_step=0
    if a.resume:
        saved=torch.load(a.resume,map_location='cpu',weights_only=True)
        assert saved['contract_sha256']==net.contract_sha256 and saved['local_fit_only'] and not saved['quantization_tested']
        net.load_state_dict(saved['model'])
        if not a.reset_optimizer:opt.load_state_dict(saved['optimizer'])
        start_step=saved['step']
        assert start_step<a.steps
    initial=copy.deepcopy(net.state_dict());initial_opt=copy.deepcopy(opt.state_dict())
    out=Path(a.out);out.mkdir(exist_ok=False)
    paths=sorted(set(executed_sources(type(net))[0]+[Path(__file__),Path(__file__).parents[1]/'model_b_contract.json']))
    for p in paths:shutil.copy2(p,out/p.name)
    frozen={str(p):file_hash(p) for p in paths}
    write_json(out/'config.json',dict(args=vars(a),specs=specs,code=frozen,
        data_sha256=file_hash(a.data),local_fit_only=True,actual_training_input='raw source',
        fresh_contract_graph=True,quantization_tested=False,extra_deployment_operators=None,
        resume_sha256=file_hash(a.resume) if a.resume else None,precision='FP32 no AMP/TF32',
        bounded_reference_sha256=reference_hashes,optimizer_reset=a.reset_optimizer,
        cross_device_resume_is_not_bitwise_equivalence_claim=True))
    def update(step):
        net.train();opt.zero_grad();p=net(x)
        if a.loss=='pixel_l1':loss=(cut(p)-cut(t)).abs().mean()
        elif a.loss=='pixel_mse':loss=(cut(p)-cut(t)).square().mean()
        elif a.loss=='pixel_sparse':loss=sparse_pixel_loss(cut(p),cut(t))
        else:
            loss,values=objective(cut(p),cut(t),cut(lab));curve=contour_coherence(cut(p),cut(t));loss=loss+.5*curve
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True)
        assert torch.isfinite(loss) and grad>0
        for g in opt.param_groups:g['lr']=.0003+.0007*.5*(1+math.cos(math.pi*step/a.steps))
        opt.step();return float(loss.detach()),float(grad)
    update(start_step);torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict()),out/'restore_probe.pt')
    expected=update(start_step+1);expected_state=copy.deepcopy(net.state_dict())
    disk=torch.load(out/'restore_probe.pt',map_location='cpu',weights_only=True)
    net.load_state_dict(disk['model']);opt.load_state_dict(disk['optimizer']);actual=update(start_step+1)
    error=max(float((v-net.state_dict()[k]).abs().max()) for k,v in expected_state.items())
    assert error==0 and expected==actual
    net.load_state_dict(initial);opt.load_state_dict(initial_opt)
    write_json(out/'preflight.json',dict(status='PASS',device=a.device,disk_resume_error=error,gpu_tested=a.device=='cuda'))
    print('FRESH_CONTRACT_PREFLIGHT',a.device,'PASS disk_resume0',flush=True)
    history=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13);started=time.monotonic()
    for step in range(start_step,a.steps+1):
        if step%a.interval==0 or step==a.steps or step==start_step:
            assert all(file_hash(p)==sha for p,sha in frozen.items()),'Executing source changed'
            assert all(file_hash(p)==sha for p,sha in reference_hashes.items()),'Reference changed'
            with torch.no_grad():
                # Exact stats on this fixed TRAIN batch, in graph order. This
                # is a local diagnostic, not validation-stat adaptation.
                hooks=[]
                for mod in net.modules():
                    if isinstance(mod,torch.nn.BatchNorm2d):
                        def calibrate(bn,inputs):
                            z=inputs[0];bn.running_mean.copy_(z.mean((0,2,3)));bn.running_var.copy_(z.var((0,2,3),unbiased=False))
                        hooks.append(mod.register_forward_pre_hook(calibrate))
                net.eval();p=net(x)
                for h in hooks:h.remove()
                buffers={k:v.clone() for k,v in net.named_buffers()}
                trainp=net.train()(x);net.eval()
                for k,v in net.named_buffers():v.copy_(buffers[k])
                calibration_error=float((p-trainp).abs().max()*128);assert calibration_error<.01
                loss,values=objective(cut(p),cut(t),cut(lab))
                row=dict(step=step,mae=float((cut(p)-cut(t)).abs().mean()*128),score=float(loss),
                    coherent=float(values['coherent_rgb_patch']),calibration_error_byte=calibration_error)
                board=Image.new('RGB',(960,1044),'#17202a');draw=ImageDraw.Draw(board)
                for i in range(3):
                    for j,(name,z) in enumerate((('原片 TRAIN',x),('固定目标',t),('从原生合同重新学习 FP32·仅诊断',p))):
                        board.paste(m.t.r.c.rgb_image(z[i]),(j*320,i*348+28));draw.text((j*320+3,i*348+3),name,font=font,fill='white')
                board.save(out/f'diagnostic_{step:06d}.png')
                torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),step=step,
                    contract_sha256=net.contract_sha256,local_fit_only=True,quantization_tested=False,
                    visual_acceptance=False,training_provenance=dict(args=vars(a),code=frozen)),out/f'float_{step:06d}.pt')
            history.append(row);write_json(out/'history.json',history);print('FRESH_CONTRACT_LOCAL',row,flush=True)
        if step==a.steps:break
        update(step)
        if (step+1)%100==0:print('FRESH_CONTRACT_STEP',step+1,flush=True)
    write_json(out/'results.json',dict(history=history,seconds=time.monotonic()-started,local_fit_only=True,visual_acceptance=False))


if __name__=='__main__':main()
