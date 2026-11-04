"""New graph INT8/Q17 probe, with independent int64 oracle. Not B1 firmware."""
from fractions import Fraction
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from .train_region_classes import ROOT,RUN,ClassNet,class_plan,read_image,as_tensor,source_manifest
from .runtime import configure_workspace,file_hash,write_json,contract_guard
from .engine import set_reproducible
from network_ab.contract_reference import q17_multiplier
sys.path.insert(0,str(Path('E:/fpga/hw/efinity/prototypes/frame_execution')))
from residual_q17_align import encode_alignment


def ratio(x):
    f=Fraction(float(x));return [f.numerator,f.denominator]


def execute(package,x,gpu=True):
    """FP32 exact integer dots on GPU versus independent int64 NumPy CPU."""
    values={'input':x};stats={}
    rnd=lambda a:torch.sign(a)*torch.floor(torch.abs(a)+.5) if gpu else np.sign(a)*np.floor(np.abs(a)+.5)
    def rq(a,m):
        if gpu:return rnd(a.double()*m/131072).clamp(-128,127).float()
        z=a.astype(np.int64)*np.asarray(m,np.int64)
        return (np.sign(z)*((np.abs(z)+65536)//131072)).clip(-128,127).astype(np.int64)
    for n in package['plan']['nodes']:
        name=n['id'];a=values[n['inputs'][0]]
        if n['kind']=='nearest_x2':
            y=a.repeat_interleave(2,-2).repeat_interleave(2,-1) if gpu else np.repeat(np.repeat(a,2,axis=2),2,axis=3)
        elif n['kind']=='residual_add':
            b=values[n['inputs'][1]];r=package['residual'][name]
            v,u=(a,b) if r['main_scaled'] else (b,a)
            y=rq(rq(v,r['multiplier'])+u,r['post_multiplier'])
        else:
            p=package['parameters'][name];k=n['kernel'];stride=n['stride'];ci=n['input_channels'];co=n['channels']
            w=np.asarray(p['weight_oihw'],np.int64);bias=np.asarray(p['bias_int32'],np.int64)[None,:,None,None]
            m=np.asarray(p['multiplier_q17'],np.int64)[None,:,None,None]
            if gpu:
                if k==3:a=F.pad(a,(1,)*4,mode='replicate')
                dot=F.conv2d(a,torch.tensor(w,device=a.device,dtype=torch.float32),stride=stride,groups=n['groups'])
                assert torch.equal(dot,dot.round());acc=dot.double()+torch.tensor(bias,device=a.device,dtype=torch.float64)
                y=rq(acc.clamp(-(2**31),2**31-1),torch.tensor(m,device=a.device,dtype=torch.float64))
                if n['activation']=='relu':y=y.clamp_min(0)
            else:
                if k==3:a=np.pad(a,((0,0),(0,0),(1,1),(1,1)),mode='edge')
                oh=(a.shape[2]-k)//stride+1;ow=(a.shape[3]-k)//stride+1
                dot=np.zeros((len(a),co,oh,ow),np.int64)
                for yy in range(k):
                    for xx in range(k):
                        z=a[:,:,yy:yy+oh*stride:stride,xx:xx+ow*stride:stride]
                        if n['groups']==ci:dot+=z*w[:,0,yy,xx][None,:,None,None]
                        else:dot+=np.einsum('nchw,oc->nohw',z,w[:,:,yy,xx])
                y=rq((dot+bias).clip(-(2**31),2**31-1),m)
                if n['activation']=='relu':y=y.clip(0)
        values[name]=y
    return values


def main():
    configure_workspace();contract_guard();set_reproducible(2026100417)
    folder=ROOT/'runs'/RUN/'classes32';out=folder/'int8_probe';out.mkdir(exist_ok=False)
    weight=folder/'final/region_classes_fp32.pt';s=torch.load(weight,map_location='cpu',weights_only=True)
    net=ClassNet().cuda().eval();net.load_state_dict(s['model']);plan=class_plan();maxima={}
    data=json.loads((folder/'data.json').read_text(encoding='utf-8'));calibration=[r['source'] for r in data['records'] if r['position']=='current']
    with torch.no_grad():
        for path in calibration:
            values={'input':as_tensor(read_image(path))[None].cuda()}
            for n in plan['nodes']:
                a=values[n['inputs'][0]]
                if n['kind']=='nearest_x2':v=a.repeat_interleave(2,-2).repeat_interleave(2,-1)
                elif n['kind']=='residual_add':v=a+values[n['inputs'][1]]
                else:v=net.layers[n['id'].replace('.','__')](a)
                values[n['id']]=v;maxima[n['id']]=max(maxima.get(n['id'],0),float(v.abs().max()))
    scales={'input':1/128};parameters={};residual={}
    for n in plan['nodes']:
        name=n['id'];si=scales[n['inputs'][0]]
        if n['kind']=='nearest_x2':scales[name]=si;continue
        so=max(maxima[name]*1.02/127,1e-7)
        if n['kind']=='residual_add':
            sb=scales[n['inputs'][1]];common=max(si,sb);so=max(so,common*1.001)
            r=dict(main_scale=ratio(si),skip_scale=ratio(sb),common_scale=ratio(common),output_scale=ratio(so))
            residual[name]=encode_alignment(r);scales[name]=so;continue
        m=net.conv(name);w=m.weight.detach().cpu().double().numpy();b=m.bias.detach().cpu().double().numpy()
        sw=np.maximum(np.abs(w).max(axis=(1,2,3))/127,1e-9);sw=np.maximum(sw,so/(si*65536));sw=np.maximum(sw,np.abs(b)/(si*2**29));so=max(so,float((si*sw).max())*1.001)
        q=lambda z:(np.sign(z)*np.floor(np.abs(z)+.5)).astype(np.int64)
        wi=q(w/sw[:,None,None,None]).clip(-127,127);bi=q(b/(si*sw));assert np.abs(bi).max()<2**31
        mult=[q17_multiplier(Fraction(si),Fraction(float(a)),Fraction(so)) for a in sw];assert all(0<m<131072 for m in mult)
        parameters[name]=dict(weight_oihw=wi.tolist(),bias_int32=bi.tolist(),multiplier_q17=mult,input_scale=ratio(si),output_scale=ratio(so),weight_scales=[ratio(v) for v in sw]);scales[name]=so
    package=dict(schema='research-region-class-int8-q17-v1',plan=plan,parameters=parameters,residual=residual,palette=s['palette'],
        fp32_sha256=file_hash(weight),calibration=calibration,production=False,output='Signed INT8 32 logits, SAME scale across classes; argmax lowest-index tie ->palette.',
        limitation='Different graph/output ABI. No RTL or physical resource/timing validation.')
    write_json(out/'runtime_parameters.json',package)
    rng=np.random.default_rng(20261004);tests=[]
    arrays=[rng.integers(-128,128,(1,3,8,12),dtype=np.int64),np.full((1,3,8,8),127,np.int64),np.full((1,3,8,8),-128,np.int64),
            (read_image(calibration[0])[:12,:16].astype(np.int64)-128).transpose(2,0,1)[None]]
    for i,a in enumerate(arrays):
        cpu=execute(package,a,False);gpu=execute(package,torch.tensor(a,device='cuda',dtype=torch.float32));errs={}
        for k,v in gpu.items():errs[k]=int(np.abs(v.cpu().numpy().astype(np.int64)-cpu[k]).max());assert errs[k]==0,(i,k,errs[k])
        tests.append(dict(index=i,shape=list(a.shape),node_errors=errs))
    palette=np.array(s['palette'],np.uint8);records=[]
    with torch.no_grad():
        for r in source_manifest()['validation']:
            a=read_image(r['source']);x=as_tensor(a)[None].cuda();fp=net(x).argmax(1)[0].cpu().numpy()
            q=execute(package,torch.tensor(a.transpose(2,0,1)[None].astype(np.float32)-128,device='cuda'))['rgb']
            pred=q.argmax(1)[0].cpu().numpy();image=palette[pred];Image.fromarray(image).save(out/(r['id']+'.png'))
            records.append(dict(id=r['id'],label_difference=float((pred!=fp).mean()),rgb_mae=float(np.abs(image.astype(float)-palette[fp]).mean())))
    write_json(out/'verification.json',dict(status='INTEGER_ARITHMETIC_ORACLE_PASS_NOT_STYLE_OR_DEPLOYMENT',tests=tests,records=records,
        runtime_sha256=file_hash(out/'runtime_parameters.json'),qat=False,calibration_frames=len(calibration),code_sha256=file_hash(__file__)))
    print('CLASS_INT8',json.dumps(records),flush=True)


if __name__=='__main__':main()
