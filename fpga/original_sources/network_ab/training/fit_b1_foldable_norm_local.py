"""Controlled CPU local-fit test of foldable normalization; not delivery."""
import argparse,copy,json,shutil,time
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from .b1_foldable_training_norm import FoldableTrainingNorm
from .probe_b1_encoder_gauge import load
from .b1_geometry_first_objective import geometry_first
from .b1_relative_rgb_profiles import relative_profiles
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','mining','out'):ap.add_argument('--'+k,required=True)
    ap.add_argument('--steps',type=int,default=1600);ap.add_argument('--interval',type=int,default=400)
    a=ap.parse_args();torch.set_num_threads(2);torch.manual_seed(2026100810)
    state,base=load(a.checkpoint);base.quantized=False
    rows={r['id']:r for r in m.load_data(a.data)['training']}
    mining=json.loads(Path(a.mining).read_text(encoding='utf-8'))['records'];specs=[mining[k] for k in (1,5,6)]
    xs=[];ts=[];labs=[]
    for spec in specs:
        row=rows[spec['row']['id']];xx,yy=spec['xy'];w,h=spec['wh']
        for key,dest in (('path',xs),('target',ts)):
            dest.append(torch.from_numpy(m.t.r.read(row[key])[yy:yy+h,xx:xx+w].transpose(2,0,1).copy()).float()/128-1)
        labs.append(torch.from_numpy(m.load_region(row['region'])[yy:yy+h,xx:xx+w].astype(np.int64)))
    x,t,lab=torch.stack(xs),torch.stack(ts),torch.stack(labs);cut=lambda z:z[...,40:-40,40:-40]
    net=FoldableTrainingNorm(base,x).eval()
    with torch.no_grad():
        original=base(x,all_nodes=True);initial=net(x,all_nodes=True);folded=net.fold()(x,all_nodes=True)
        before={k:float((v-initial[k]).abs().max()) for k,v in original.items()}
        after={k:float((v-folded[k]).abs().max()) for k,v in initial.items()}
        assert max(before.values())<.05 and max(after.values())<.05,(before,after)
    out=Path(a.out);out.mkdir(exist_ok=False)
    paths=[Path(__file__),Path(__file__).with_name('b1_foldable_training_norm.py')]
    for p in paths:shutil.copy2(p,out/p.name)
    write_json(out/'config.json',dict(args=vars(a),specs=specs,checkpoint_sha256=file_hash(a.checkpoint),
        code={str(p):file_hash(p) for p in paths},initial_warm_float_errors=before,initial_fold_float_errors=after,
        local_fit_only=True,training_data_only=True,batch_norm_deployment=False))
    params=[v for v in net.parameters() if v.requires_grad]
    opt=torch.optim.Adam(params,lr=.0003)
    def update():
        net.train();opt.zero_grad();p=net(x)
        loss,_=geometry_first(cut(p),cut(t),cut(lab));relative,_=relative_profiles(cut(p),cut(t),cut(lab))
        loss=loss+relative;loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(params,5,error_if_nonfinite=True);assert grad>0
        opt.step();return float(loss.detach()),float(grad)
    # Actual disk restore of parameters, running buffers and Adam state.
    update();torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict()),out/'restore_probe.pt')
    expected=update();expected_state=copy.deepcopy(net.state_dict())
    disk=torch.load(out/'restore_probe.pt',map_location='cpu',weights_only=True)
    net.load_state_dict(disk['model']);opt.load_state_dict(disk['optimizer']);got=update()
    err=max(float((v-net.state_dict()[k]).abs().max()) for k,v in expected_state.items())
    assert err==0 and expected==got
    # Start the test from its original controlled warm function.
    net=FoldableTrainingNorm(base,x).eval();params=[v for v in net.parameters() if v.requires_grad]
    opt=torch.optim.Adam(params,lr=.0003);history=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13);start=time.monotonic()
    write_json(out/'preflight.json',dict(device='cpu',disk_resume_error=err,status='PASS',gpu_training=False))
    print('FOLDABLE_NORM_CPU_PREFLIGHT PASS warm/fold/disk_resume0',flush=True)
    for step in range(a.steps+1):
        if step%a.interval==0 or step==a.steps:
            with torch.no_grad():
                net.calibrate_running(x);p=net(x);fold=net.fold();q=fold(x)[0]
                fold_error=float((p-q).abs().max()*128);assert fold_error<.05
                fold.calibrate();fold.quantized=True;integer=fold(x)[0]
                row=dict(step=step,float_rgb_mae=float((cut(p)-cut(t)).abs().mean()*128),
                    same_weights_quantized_mae=float((cut(integer)-cut(t)).abs().mean()*128),fold_error_byte=fold_error)
                board=Image.new('RGB',(960,1044),'#17202a');draw=ImageDraw.Draw(board)
                for i in range(3):
                    for j,(name,z) in enumerate((('目标',t),('归一化后折叠FP32·局部诊断',p),('直接INT8·未经QAT',integer))):
                        board.paste(m.t.r.c.rgb_image(z[i]),(j*320,i*348+28));draw.text((j*320+3,i*348+3),name,font=font,fill='white')
                board.save(out/f'diagnostic_{step:06d}.png')
                saved=copy.deepcopy(state);saved.update(model=fold.state_dict(),step=step,local_fit_only=True,visual_acceptance=False,
                    norm_fold_diagnostic=True,qat=True,training_provenance=dict(kind='local FP32 foldable BatchNorm diagnostic; direct untrained INT8 conversion',data=a.data))
                torch.save(saved,out/f'folded_{step:06d}.pt')
            history.append(row);write_json(out/'history.json',history);print('FOLDABLE_NORM_LOCAL',row,flush=True)
        if step==a.steps:break
        update()
    write_json(out/'results.json',dict(history=history,seconds=time.monotonic()-start,local_fit_only=True,visual_acceptance=False))


if __name__=='__main__':main()
