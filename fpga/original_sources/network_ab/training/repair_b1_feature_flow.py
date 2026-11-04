"""Controlled general-scene repair of five disconnected stem feature lanes."""
import copy
import json
import math
import shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible
from .train_b1_deep_contour import DeepTraining,OUT as LABELS
from . import resume_b1_contour_precision as general
from .probe_b1_crop_quantization import calibrate_parameter_precision
from .probe_b1_single_crop_fit import inputs,objective
from .b1_trained_image import runtime_from_image
from .probe_b1_cartoon_int8 import IntegerB1

OUT=LABELS.parent/'b1-feature-flow-repair-20261005-v1'
PARENT=general.OUT/'checkpoint_002000.pt'


def reconnect(net):
    assert float(net.weight['down1__pw'][:,3:8].detach().abs().max())==0
    assert float(net.weight['down1__dw'][3:8].detach().abs().max())==0
    with torch.no_grad():
        net.weight['down1__dw'][3:8].zero_()
        net.weight['down1__dw'][3:8,0,1,1]=112
        net.bias['down1__dw'][3:8].zero_()
        net.down1__dw_m[:,3:8]=1170
        for kind,parameters in [('w',net.weight),('b',net.bias)]:
            getattr(net,kind+'_down1__dw_initial')[3:8].copy_(parameters['down1__dw'][3:8])
        net.package['parameters']['down1.dw']['multipliers']=net.down1__dw_m.flatten().long().tolist()


def make(data,fixed):
    tr=DeepTraining(data);s=torch.load(PARENT,map_location='cpu',weights_only=True)
    tr.net.load_state_dict(s['model']);tr.net.package=s['runtime'];tr.net.quantized=False
    if fixed:reconnect(tr.net)
    tr.opt=torch.optim.Adam([p for p in tr.net.parameters() if p.requires_grad],lr=.025)
    return tr


def gradients(net,x,t):
    net.zero_grad(set_to_none=True);nodes=net(x,True,True)
    objective(nodes['res2_add'][:,24:25],t).backward()
    return dict(stem_feature_grad=float(net.weight['stem'].grad[3:8].norm()),
        depthwise_feature_grad=float(net.weight['down1__dw'].grad[3:8].norm()),
        downstream_feature_column_grad=float(net.weight['down1__pw'].grad[:,3:8].norm()),
        feature_max=float(nodes['down1.dw'][:,3:8].detach().abs().max()))


@torch.no_grad()
def clutter_audit(data):
    arms={key:IntegerB1(runtime_from_image((OUT/(key+'_001000')/'coefficients-padded.bin').read_bytes())) for key in ('control','reconnected')}
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    board=Image.new('RGB',(1680,314*len(general.gate.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    stats=[];images={}
    for r in data:
        if r['split']=='train':continue
        src=np.array(Image.open(r['source']));x=torch.tensor(src.transpose(2,0,1)[None].copy(),device='cuda',dtype=torch.float32)-128
        t=np.maximum(np.load(r['label']).astype(np.float32)[::4,::4]*32-8,0)
        valid=np.zeros_like(t,dtype=bool);valid[8:-8,8:-8]=True
        for key,net in arms.items():
            nodes,_=net(x,True);p=nodes['res3_conv1.dw'][0,24].cpu().numpy()
            empty=(t==0)&valid
            stats.append(dict(id=r['id'],arm=key,background_response_gt0=float(((p>0)&empty).sum()/max(empty.sum(),1)),
                background_response_gt4=float(((p>4)&empty).sum()/max(empty.sum(),1)),
                missed_boundary_gt4=float(((t>4)&(p<=4)&valid).sum()/max(((t>4)&valid).sum(),1))))
            if key=='reconnected':
                overlay=np.zeros((*p.shape,3),np.uint8)
                overlay[(p>0)&(t>0)]=[180,180,180]
                overlay[(p>0)&(t==0)]=[255,75,65]
                overlay[(t>4)&(p<=4)]=[0,200,255]
                images[(r['id'],'errors')]=Image.fromarray(overlay).resize((1280,720),Image.Resampling.NEAREST)
            images[(r['id'],key)]=general.gray(p)
        images[(r['id'],'target')]=Image.open(LABELS/'labels'/(r['id']+'_thin.png')).convert('RGB')
    for i,(key,box,_) in enumerate(general.gate.CROPS):
        for j,(col,title) in enumerate([('target','细线标签'),('control','对照：旧连接继续训'),('reconnected','修复：实际权重重载'),('errors','红：杂响应；蓝：漏边')]):
            board.paste(images[(key,col)].crop(box),(420*j,314*i+34));draw.text((420*j+4,314*i+4),title,font=font,fill='white')
    board.save(OUT/'native_clutter_audit.png');write_json(OUT/'clutter_metrics.json',stats)


def main():
    configure_workspace();set_reproducible(2026100511);contract_guard();OUT.mkdir(exist_ok=False);general.OUT=OUT
    data=json.loads((LABELS/'data.json').read_text(encoding='utf-8'))
    for r in data:
        for key in ('source','target','label'):assert file_hash(r[key])==r[key+'_sha256']
    write_json(OUT/'binding.json',dict(parent=str(PARENT),parent_sha256=file_hash(PARENT),
        data_sha256=file_hash(LABELS/'data.json'),fp32_steps=600,qat_steps=400,
        scope='Only reconnect dormant stem lanes; unchanged labels and RGB branch, matched training order'))
    for name in ('repair_b1_feature_flow.py','train_b1_structural_features.py','resume_b1_contour_precision.py','train_b1_deep_contour.py'):
        shutil.copy2(Path(__file__).with_name(name),OUT/name)
    control=make(data,False);fixed=make(data,True);_,_,_,x,t=inputs()
    with torch.no_grad():
        difference=float((control.net(x)-fixed.net(x)).abs().max());assert difference==0
        control.net.quantized=fixed.net.quantized=True
        integer_difference=float((control.net(x)-fixed.net(x)).abs().max());assert integer_difference==0
        control.net.quantized=fixed.net.quantized=False
    proof=dict(before=gradients(control.net,x,t),reconnected=gradients(fixed.net,x,t),
        initial_fp32_output_difference=difference,initial_integer_output_difference=integer_difference)
    assert proof['before']['feature_max']==0 and proof['before']['stem_feature_grad']==0
    assert proof['reconnected']['downstream_feature_column_grad']>0
    fixed.update();proof['after_one_update']=gradients(fixed.net,x,t)
    assert proof['after_one_update']['stem_feature_grad']>0
    # Native forward/backward and fresh-instance checkpoint restoration.
    proof['native_update']=fixed.update(True)
    general.save(fixed,OUT/'resume_probe.pt');expected_record=fixed.update()
    expected={k:v.clone() for k,v in fixed.net.state_dict().items()}
    restored=make(data,True);general.restore(restored,OUT/'resume_probe.pt');got=restored.update()
    error=max(float((v-expected[k]).abs().max()) for k,v in restored.net.state_dict().items())
    assert error==0 and got['crop']==expected_record['crop'];proof['disk_resume_error']=error
    write_json(OUT/'preflight.json',proof);print('FEATURE_RECONNECT_PROVED',proof,flush=True)
    del control,fixed,restored;torch.cuda.empty_cache()
    arms={'control':make(data,False),'reconnected':make(data,True)}
    with (OUT/'metrics.jsonl').open('x',encoding='utf-8',buffering=1) as f:
        for step in range(1,1001):
            reports={}
            for key,tr in arms.items():
                if step==601:
                    changes=calibrate_parameter_precision(tr.net);write_json(OUT/(key+'_calibration.json'),changes)
                    tr.net.quantized=True;tr.opt=torch.optim.Adam([p for p in tr.net.parameters() if p.requires_grad],lr=.01)
                lr=(.025-.01*(step/600)) if step<=600 else .002+.008*.5*(1+math.cos(math.pi*(step-601)/400))
                for group in tr.opt.param_groups:group['lr']=lr
                reports[key]=tr.update(step%100==0)
            assert reports['control']['sample']==reports['reconnected']['sample'] and reports['control']['crop']==reports['reconnected']['crop']
            f.write(json.dumps(dict(step=step,reports=reports))+'\n')
            if step%100==0:print('FEATURE_FLOW',step,{k:round(v['loss'],5) for k,v in reports.items()},flush=True)
            if step in (600,1000):
                for key,tr in arms.items():
                    general.save(tr,OUT/f'{key}_{step:06d}.pt')
                    general.review(tr,data,f'{key}_{step:06d}',tr.net.quantized)
    clutter_audit(data)
    print('FEATURE_FLOW_COMPLETE',flush=True)


if __name__=='__main__':main()
