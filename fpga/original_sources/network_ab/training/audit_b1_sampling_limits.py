"""Read-only sampling investigation: controls are diagnostics, never delivery.

No training or hardware changes. Distinguish outer display scaling, B1's spatial
bottleneck, current learned representation, and final nineteen-level decoding.
"""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_learned_colour as c
from . import train_b1_19_levels as n
from . import train_b1_19_crisp as crisp
from .b1_trained_image import Parameters,encode,decode
from .engineer_b1_blocks import oracle_records
from .fpga_video_resize import area_3to2,bilinear_2to3,verify
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from network_ab.contract_reference import execute_integer

OUT=crisp.OUT/'sampling-limit-audit'
SOURCE=crisp.OUT/'qat/package/coefficients-padded.bin'


def identity_image():
    _,plan=contract_guard();params={}
    gauss=np.outer([1,2,1],[1,2,1])/16
    aligned=np.array([[0,0,0],[0,.25,.25],[0,.25,.25]])
    for r in plan['nodes']:
        if not r['kernel']:continue
        name=r['id'];ci=1 if r['kind']=='dw3x3' else r['input_channels']
        w=np.zeros((r['channels'],ci,r['kernel'],r['kernel']),np.float64);b=np.zeros(len(w))
        if name=='stem':
            for k in range(len(w)):w[k,k%3,1,1]=.5;b[k]=64
        elif name=='rgb_head':
            for k in range(3):w[k,k,0,0]=2;b[k]=-128
        elif name.startswith('res'):
            pass # Zero residual branch, identity main branch.
        elif r['kind']=='dw3x3':
            if name in ('down1.dw','down2.dw'):w[:,0]=gauss
            elif name in ('up1_high_dw','up2_high_dw'):w[:,0]=aligned
            else:w[:,0,1,1]=1
        else:
            for k in range(len(w)):w[k,k%3,0,0]=1
        wi=np.rint(w*32).astype(np.int64);bi=np.rint(b*32).astype(np.int64)
        assert abs(wi).max()<=127
        flat=wi.transpose(0,2,3,1).reshape(len(w),-1)
        params[name]=Parameters(tuple(map(tuple,flat.tolist())),tuple(bi.tolist()),tuple([4096]*len(w)),r['activation']=='relu')
    residual={f'res{i}_add':dict(main_scaled=True,multiplier=131071,post_multiplier=131071) for i in range(1,4)}
    raw=encode(params,residual);return raw+bytes(20480-len(raw))


def tensor(a):return torch.tensor(a.transpose(2,0,1)[None].copy(),device='cuda',dtype=torch.float32)-128
def rgb(z):return c.rgb_image(z[0]/128)


def conv(net,x,key):
    w,b=net.weight[key],net.bias[key];m=getattr(net,key+'_m');scale=(m.flatten()/131072).float()
    wi=net.round(w/scale[:,None,None,None]).clamp(-127,127);bi=net.round(b/scale).double()[None,:,None,None]
    if key=='up2_high_dw':x=F.pad(x,(1,1,1,1),mode='replicate')
    dot=F.conv2d(x,wi,groups=8 if key=='up2_high_dw' else 1)
    y=net.round((dot.double()+bi)*m/131072).clamp(-128,127).float()
    return y.clamp_min(0) if key=='up2_high_dw' else y


def panel(rows,titles,path,width=420,height=280):
    board=Image.new('RGB',(width*len(titles),(height+34)*len(rows)),'#17202a');d=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    for i,images in enumerate(rows):
        for j,img in enumerate(images):
            assert img.size==(width,height)
            board.paste(img,(j*width,i*(height+34)+34));d.text((j*width+4,i*(height+34)+4),titles[j],font=font,fill='white')
    board.save(path)


@torch.inference_mode()
def main():
    root=configure_workspace();contract,plan=contract_guard();cv2.setNumThreads(2);torch.set_num_threads(4)
    OUT.mkdir(exist_ok=False);source_hash=file_hash(SOURCE)
    actual=c.IntegerB1(c.runtime_from_image(SOURCE.read_bytes()));net=c.ColourB1(False,SOURCE).cuda().eval();net.quantized=True
    raw=identity_image();(OUT/'diagnostic_identity_not_style.bin').write_bytes(raw);identity=c.IntegerB1(c.runtime_from_image(raw))
    pp,rr=decode(raw);op,rp=oracle_records(pp);sample=np.random.default_rng(614).integers(-128,128,(12,12,3),dtype=np.int16)
    reference=execute_integer(contract,sample.tolist(),op,rp)
    nodes,_=identity(torch.tensor(sample.transpose(2,0,1)[None].copy(),device='cuda',dtype=torch.float32),True)
    oracle_error=max(float(np.abs(z[0].permute(1,2,0).cpu().numpy()-np.array(reference['tensors'][key])).max()) for key,z in nodes.items());assert oracle_error==0
    # Periodic details cannot be reconstructed from a spatially constant latent
    # by NN-copy + translation-equivariant convolutions, regardless of weights.
    periodic=[];pattern_rows=[];trace_names=['input','stem','down1.dw','down1.pw','down2.dw','down2.pw','res3_add','up1_low_pw','up1_nn','up2_low_pw','up2_nn','rgb_head']
    for period in (2,4,8,16):
        yy,xx=np.indices((256,256));a=np.repeat(np.where((xx%(period))<period//2,64,192).astype(np.uint8)[:,:,None],3,2)
        q=net(tensor(a)/128,all_nodes=True);trace=[]
        for name in trace_names:
            z=q[name];h,w=z.shape[-2:];v=z[:,:,h//4:3*h//4,w//4:3*w//4]
            trace.append(dict(node=name,shape=list(z.shape[1:]),max_channel_spatial_range=float((v.amax((-2,-1))-v.amin((-2,-1))).max())))
        if period<=4:
            assert trace[-1]['max_channel_spatial_range']==0
            assert next(r for r in trace if r['node']=='down2.dw')['max_channel_spatial_range']==0
        periodic.append(dict(period_720=period,input_contrast=128,trace=trace))
        im=rgb(q['rgb_head']);pattern_rows.append([Image.fromarray(a).crop((64,64,192,192)).resize((384,384),Image.Resampling.NEAREST),im.crop((64,64,192,192)).resize((384,384),Image.Resampling.NEAREST)])
        print('PERIOD_TEST',period,'output_range',trace[-1]['max_channel_spatial_range'],flush=True)
    panel(pattern_rows,['输入条纹（诊断放大3倍）','当前B1输出（诊断放大3倍）'],OUT/'periodic_controls.png',384,384)
    data=json.loads((c.LABELS/'data.json').read_text(encoding='utf-8'));boards=[];oracleboards=[];natural=[];node_shapes=None
    for key,box,label in c.CROPS:
        row=next(r for r in data if r['id']==key)
        for name in ('source','target'):assert file_hash(row[name])==row[name+'_sha256']
        a=np.array(Image.open(row['source']).convert('RGB'));t=c.image_tensor(row['target'])[None];x=tensor(a)
        z=net(x/128,all_nodes=True);assert torch.equal(z['rgb_head'],actual(x))
        if node_shapes is None:node_shapes={k:list(v.shape[1:]) for k,v in z.items()}
        passthrough=identity(x);boards.append([Image.fromarray(a).crop(box),rgb(passthrough).crop(box),rgb(z['rgb_head']).crop(box)])
        labels=n.Trainer.make_targets(None,t)
        gt=labels.clone();gt[:,:6]=gt[:,:6].mul(3/127).round();gt[:,6:]=net.round(gt[:,6:]/2)
        direct=conv(net,gt,'rgb_head')
        lattice=labels[:,:,::2,::2].repeat_interleave(2,-2).repeat_interleave(2,-1)
        sampled=conv(net,conv(net,lattice,'up2_high_dw'),'rgb_head')
        oracleboards.append([c.rgb_image(t[0]).crop(box),rgb(direct).crop(box),rgb(sampled).crop(box),rgb(z['rgb_head']).crop(box)])
        natural.append(dict(id=key,box=list(box),rgb_mae_identity=float((passthrough-x).abs().mean()),
            target_representation_mae=float((direct-t*128).abs().mean()),oracle360_mae=float((sampled-t*128).abs().mean()),
            actual_mae=float((z['rgb_head']-t*128).abs().mean())))
    panel(boards,['720p原输入','同B1采样：诊断直通权重','当前B1风格权重'],OUT/'same_graph_controls.png')
    panel(oracleboards,['教师目标','仅诊断：19层直接解码','仅诊断：正确360格点→末端','实际小网络'],OUT/'decoder_oracles.png')
    clips=json.loads((root/'runs/clear-style-control-20261003/video_clips.json').read_text(encoding='utf-8'))['clips'];outer=[];outerrows=[]
    for key,frame,box in [('stairs',90,(780,70,1200,350)),('plaza',210,(300,370,720,650))]:
        clip=next(r for r in clips if r['id']==key);assert file_hash(clip['clip'])==clip['clip_sha256']
        cap=cv2.VideoCapture(clip['clip']);cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,bgr=cap.read();cap.release();assert ok
        full=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);low=area_3to2(full);back=bilinear_2to3(low)
        pred=np.array(rgb(actual(tensor(low))));styled=bilinear_2to3(pred);b=tuple(int(v*1.5) for v in box)
        outerrows.append([Image.fromarray(a).crop(b) for a in (full,back,styled)])
        delta=np.abs(full.astype(np.int16)-back.astype(np.int16));outer.append(dict(scene=key,frame=frame,source=clip['clip'],sha256=clip['clip_sha256'],resize_mae=float(delta.mean()),resize_p99=float(np.percentile(delta,99))))
    panel(outerrows,['1080p原帧','仅1080→720→1080，无CNN','再加入当前B1风格网络'],OUT/'outer_resize_controls.png',630,420)
    budget={k:dict(shape_chw=v,scalar_count=int(np.prod(v)),max_stored_bits=int(np.prod(v))*8) for k,v in node_shapes.items() if k in ('input','stem','down1.dw','down1.pw','down2.dw','down2.pw','up1_low_pw','rgb_head')}
    assert file_hash(SOURCE)==source_hash
    write_json(OUT/'results.json',dict(weights=str(SOURCE),weights_sha256=source_hash,contract_sha256=file_hash(Path(c.__file__).parents[1]/'model_b_contract.json'),
        unchanged_source_package=True,training_steps_added=0,hardware_changes=False,actual_board=False,
        resize_verification=verify(),identity_same_graph_oracle_error=oracle_error,activation_capacity=budget,
        periodic=periodic,natural=natural,outer_resize=outer,
        caveats=['Identity is a constructed same-graph diagnostic, not a trained upper bound or style candidate.',
                 'Correct360-label injection bypasses encoder and most decoder; does not prove180 bottleneck lossless.',
                 'Periodic texture failure proves a limitation for those patterns, not that every contour must be blurred.',
                 'INT8 state-count bound applies to arbitrary RGB arrays, not a natural-image perceptual compression bound.']))
    print('SAMPLING_LIMIT_AUDIT_COMPLETE',json.dumps(natural),json.dumps(outer),flush=True)


if __name__=='__main__':main()
