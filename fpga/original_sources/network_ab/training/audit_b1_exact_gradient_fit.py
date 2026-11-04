"""FP32 exact-gradient/local optimizer diagnostic, not deployable output.

Use ColourB1 directly: no training-only leaky derivatives, no rounding. The
purpose is to test a stronger local solver before blaming graph capacity.
Both FP32 and the actual quantized replay are explicitly measured.
"""
import argparse,copy,json,shutil,time
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .b1_geometry_first_objective import geometry_first
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','mining','out'):ap.add_argument('--'+k,required=True)
    ap.add_argument('--iterations',type=int,default=200);ap.add_argument('--device',default='cpu')
    a=ap.parse_args();torch.set_num_threads(2);torch.manual_seed(1083)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    state,net=load(a.checkpoint);net.to(a.device);net.quantized=False;net.requires_grad_(True)
    assert not hasattr(net,'training_activation_proxy')
    rows={r['id']:r for r in m.load_data(a.data)['training']}
    mined=json.loads(Path(a.mining).read_text(encoding='utf-8'))['records'];specs=[mined[i] for i in (1,5,6)]
    xs=[];ts=[];ls=[]
    for spec in specs:
        row=rows[spec['row']['id']];xx,yy=spec['xy'];w,h=spec['wh']
        def read(k):return torch.from_numpy(m.t.r.read(row[k])[yy:yy+h,xx:xx+w].transpose(2,0,1).copy()).float().to(a.device)/128-1
        xs.append(read('path'));ts.append(read('target'));ls.append(torch.from_numpy(m.load_region(row['region'])[yy:yy+h,xx:xx+w].astype(np.int64)).to(a.device))
    x,t,lab=torch.stack(xs),torch.stack(ts),torch.stack(ls);cut=lambda z:z[...,40:-40,40:-40]
    out=Path(a.out);out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name)
    opt=torch.optim.LBFGS(net.parameters(),lr=.5,max_iter=1,max_eval=10,tolerance_grad=1e-8,tolerance_change=1e-10,history_size=30,line_search_fn='strong_wolfe')
    calls=0;history=[];start=time.monotonic();font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13)
    def closure():
        nonlocal calls
        opt.zero_grad();p=net(x)[0];loss,_=geometry_first(cut(p),cut(t),cut(lab));loss.backward()
        assert torch.isfinite(loss) and all(v.grad is not None and torch.isfinite(v.grad).all() for v in net.parameters())
        calls+=1;return loss
    # A small directional finite-difference check against the true FP32
    # objective. Nonsmooth activation crossings are avoided with tiny steps.
    l0=closure();k='rgb_head';w=net.weight[k];flat=w.grad.flatten();at=int(flat.abs().argmax());g=float(flat[at]);old=float(w.flatten()[at]);eps=1e-5
    with torch.no_grad():
        values=[]
        for direction in (-1,1):
            w.flatten()[at]=old+direction*eps;p=net(x)[0];values.append(float(geometry_first(cut(p),cut(t),cut(lab))[0]))
        w.flatten()[at]=old
    fd=(values[1]-values[0])/(2*eps)
    write_json(out/'gradient_probe.json',dict(parameter=k,index=at,autograd=g,finite_difference=fd,epsilon=eps,
        limitation='One strong RGB-head coordinate; no whole-network numerical-gradient claim.'))
    assert g*fd>0 and abs(g-fd)<max(.05,.15*abs(g)),(g,fd)
    for step in range(a.iterations+1):
        if step%25==0 or step==a.iterations:
            with torch.no_grad():
                p=net(x)[0];loss,_=geometry_first(cut(p),cut(t),cut(lab));net.quantized=True;q=net(x)[0];net.quantized=False
                limits={key:float((v.abs()>127*(getattr(net,key+'_m')/131072).float().flatten()[:,None,None,None]).float().mean()) for key,v in net.weight.items()}
                row=dict(step=step,function_calls=calls,loss=float(loss),float_mae=float((cut(p)-cut(t)).abs().mean()*128),
                    quantized_mae=float((cut(q)-cut(t)).abs().mean()*128),outside_coefficient_range=limits)
                board=Image.new('RGB',(960,1044),'#17202a');draw=ImageDraw.Draw(board)
                for i in range(3):
                    for j,(name,z) in enumerate((('目标',t),('FP32精确梯度·仅诊断',p),('直接量化对照·未校准',q))):
                        board.paste(m.t.r.c.rgb_image(z[i]),(320*j,348*i+28));draw.text((320*j+3,348*i+3),name,font=font,fill='white')
                board.save(out/f'diagnostic_{step:06d}.png')
            history.append(row);write_json(out/'history.json',history);print('EXACT_GRADIENT_FIT',row,flush=True)
            saved=copy.deepcopy(state);saved.update(model=net.state_dict(),optimizer=opt.state_dict(),step=step,qat=False,
                local_fit_only=True,exact_gradient_fp32_diagnostic=True,visual_acceptance=False)
            torch.save(saved,out/f'diagnostic_{step:06d}.pt')
        if step==a.iterations:break
        opt.step(closure)
    write_json(out/'results.json',dict(history=history,seconds=time.monotonic()-start,checkpoint_sha256=file_hash(a.checkpoint),
        source_sha256=file_hash(__file__),specs=specs,not_deployable=True,local_fit_not_generalization=True))


if __name__=='__main__':main()
