"""Second-order local fitting after the plain native-contract FP32 control.

This is not the old byte-scaled ColourB1 L-BFGS experiment. It starts from
the saved native Conv/BN graph, freezes its TRAIN-calibrated BN statistics,
uses smooth pixel regression, and has no custom loss/STE/proxy/decoder.
Local TRAIN-only diagnostic; no quantization or deployment claim.
"""
import argparse,copy,json,shutil,time
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from network_ab.contract_torch import build_fp32_model
from .runtime import configure_workspace,file_hash,write_json
from .engine import set_reproducible
from .train_b1_late_region_supervision import executed_sources
from . import train_b1_minimum_area as m


def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','mining','out'):ap.add_argument('--'+k,required=True)
    ap.add_argument('--steps',type=int,default=500)
    a=ap.parse_args();configure_workspace();set_reproducible(2026100815);torch.set_num_threads(2)
    data=m.load_data(a.data);byid={r['id']:r for r in data['training']}
    specs=json.loads(Path(a.mining).read_text(encoding='utf-8'))['records'];specs=[specs[k] for k in (1,5,6)]
    xx=[];tt=[]
    for s in specs:
        r=byid[s['row']['id']];x,y=s['xy'];w,h=s['wh']
        for k,dst in (('path',xx),('target',tt)):
            dst.append(torch.from_numpy(m.t.r.read(r[k])[y:y+h,x:x+w].transpose(2,0,1).copy()).float()/128-1)
    x,t=torch.stack(xx).cuda(),torch.stack(tt).cuda();cut=lambda z:z[...,40:-40,40:-40]
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True);net=build_fp32_model().cuda().eval()
    assert state['local_fit_only'] and not state['quantization_tested'] and state['contract_sha256']==net.contract_sha256
    net.load_state_dict(state['model']);initial=copy.deepcopy(net.state_dict())
    params=list(net.parameters())
    def optimizer():return torch.optim.LBFGS(params,lr=1,max_iter=1,max_eval=12,history_size=30,
        tolerance_grad=1e-9,tolerance_change=1e-12,line_search_fn='strong_wolfe')
    opt=optimizer();calls=0;last_grad=0
    out=Path(a.out);out.mkdir(exist_ok=False)
    paths=sorted(set(executed_sources(type(net))[0]+[Path(__file__)]));hashes={str(p):file_hash(p) for p in paths}
    for p in paths:shutil.copy2(p,out/p.name)
    write_json(out/'config.json',dict(args=vars(a),specs=specs,code=hashes,checkpoint_sha256=file_hash(a.checkpoint),
        data_sha256=file_hash(a.data),kind='local FP32 L-BFGS; fixed TRAIN BN statistics',quantization_tested=False,
        inference_extra_ops='BN still needs folding and integer verification',precision='FP32 no AMP/TF32'))
    def closure():
        nonlocal calls,last_grad
        opt.zero_grad();p=net(x);e=cut(p)-cut(t)
        loss=torch.nn.functional.smooth_l1_loss(cut(p),cut(t),beta=1/128)
        loss.backward();calls+=1
        assert torch.isfinite(loss) and all(v.grad is not None and torch.isfinite(v.grad).all() for v in params)
        last_grad=float(torch.cat([v.grad.flatten() for v in params]).norm());assert last_grad>0
        return loss
    # Test actual disk optimizer/history restore and fixed BN buffers.
    opt.step(closure);torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict()),out/'restore_probe.pt')
    expected=float(opt.step(closure));expected_state=copy.deepcopy(net.state_dict())
    disk=torch.load(out/'restore_probe.pt',map_location='cuda',weights_only=True)
    net.load_state_dict(disk['model']);opt.load_state_dict(disk['optimizer']);actual=float(opt.step(closure))
    error=max(float((v-net.state_dict()[k]).abs().max()) for k,v in expected_state.items());assert error==0 and expected==actual
    net.load_state_dict(initial);opt=optimizer();calls=0
    with torch.no_grad():baseline=net(x).clone()
    write_json(out/'preflight.json',dict(status='PASS',gpu_forward_backward=True,disk_next_step_error=error))
    print('PLAIN_LBFGS_GPU_PREFLIGHT_PASS disk0',flush=True)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13);history=[];started=time.monotonic()
    for step in range(a.steps+1):
        if step%100==0 or step==a.steps:
            assert all(file_hash(p)==v for p,v in hashes.items())
            with torch.no_grad():p=net(x)
            rec=dict(step=step,closure_calls=calls,mae=float((cut(p)-cut(t)).abs().mean()*128),last_grad=last_grad)
            history.append(rec);write_json(out/'history.json',history)
            board=Image.new('RGB',(960,1044),'#17202a');draw=ImageDraw.Draw(board)
            for i in range(3):
                for j,(title,z) in enumerate((('固定目标',t),('像素拟合8000 FP32',baseline),('二阶拟合 FP32·仅局部诊断',p))):
                    board.paste(m.t.r.c.rgb_image(z[i]),(j*320,i*348+28));draw.text((j*320+3,i*348+3),title,font=font,fill='white')
            board.save(out/f'diagnostic_{step:06d}.png')
            torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),step=step,contract_sha256=net.contract_sha256,
                local_fit_only=True,quantization_tested=False,visual_acceptance=False,training_provenance=dict(args=vars(a),code=hashes)),out/f'float_{step:06d}.pt')
            print('PLAIN_LBFGS_LOCAL',rec,flush=True)
        if step==a.steps:break
        opt.step(closure)
        assert all(torch.equal(v,initial[k]) for k,v in net.named_buffers())
    write_json(out/'results.json',dict(history=history,seconds=time.monotonic()-started,visual_acceptance=False,quantization_tested=False))


if __name__=='__main__':main()
