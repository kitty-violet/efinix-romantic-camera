"""Calibrate, encode, reload and execute actual trained B1 resident bytes."""
import argparse
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import struct
import zlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from network_ab.contract_torch import build_fp32_model
from network_ab.contract_reference import q17_multiplier, execute_integer
from .train_b1_deploy_preview import OUT, ROOT, setup
from .probe_b1_cartoon_int8 import IntegerB1, ratio, encode_alignment
from .train_b1_distillation import read_image, as_tensor, to_rgb
from .runtime import file_hash, write_json, contract_guard
from . import b1_trained_image as codec
from .fpga_video_resize import bilinear_2to3


def quant(x):return np.sign(x)*np.floor(np.abs(x)+.5)


@torch.inference_mode()
def main(step):
    cfg,data,binding=setup();contract,plan=contract_guard()
    source=OUT/f'checkpoint_{step:06d}.pt';state=torch.load(source,map_location='cpu',weights_only=True)
    assert state['binding']==binding and state['step']==step
    metrics=[json.loads(line) for line in (OUT/'metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in metrics][:step]==list(range(1,step+1))
    out=OUT/f'package-{step}';out.mkdir(exist_ok=False)
    net=build_fp32_model().cuda().eval();net.load_state_dict(state['model'])
    calibration=[r['current'] for r in data['pairs']];maxima={}
    for i,path in enumerate(calibration):
        values=net(as_tensor(read_image(path))[None].cuda(),return_tensors=True)
        for name,v in values.items():maxima[name]=max(maxima.get(name,0.),float(v.abs().max()))
        print('CALIBRATE',i+1,len(calibration),flush=True)
    folded=copy.deepcopy(net)
    for seq in folded.operators.values():
        for i,layer in enumerate(seq):
            if isinstance(layer,torch.nn.BatchNorm2d):seq[i-1]=torch.nn.utils.fuse_conv_bn_eval(seq[i-1],layer);seq[i]=torch.nn.Identity()
    scales={'input':1/128};params={};residual={};encoding={}
    for row in plan['nodes']:
        name=row['id'];si=scales[row['inputs'][0]]
        if row['kind']=='nearest_x2':scales[name]=si;continue
        if row['kind']=='residual_add':
            skip=scales[row['inputs'][1]];common=max(si,skip);so=max(maxima[name]*1.02/127,common*1.001)
            rp=dict(main_scale=ratio(si),skip_scale=ratio(skip),common_scale=ratio(common),output_scale=ratio(so))
            residual[name]=rp;encoding[name]=encode_alignment(rp);scales[name]=so;continue
        conv=next(m for m in folded.operators[folded.names[name]] if isinstance(m,torch.nn.Conv2d))
        w=conv.weight.cpu().double().numpy();b=conv.bias.cpu().double().numpy()
        so=1/128 if name=='rgb_head' else max(maxima[name]*1.02/127,1e-7)
        sw=np.maximum(np.max(np.abs(w),axis=(1,2,3))/127,1e-9)
        sw=np.maximum(sw,so/(si*65536));sw=np.maximum(sw,np.abs(b)/(si*(2**29)))
        if name!='rgb_head':so=max(so,float((si*sw).max())*1.001)
        multipliers=[q17_multiplier(Fraction(si),Fraction(float(v)),Fraction(so)) for v in sw]
        assert all(0<m<131072 for m in multipliers)
        wi=quant(w/sw[:,None,None,None]).clip(-127,127).astype(np.int32);bi=quant(b/(si*sw)).astype(np.int64)
        assert np.abs(bi).max()<2**31
        params[name]=dict(weights=wi.transpose(0,2,3,1).reshape(row['channels'],-1).tolist(),bias=bi.tolist(),multipliers=multipliers,
                          input_scale=ratio(si),output_scale=ratio(so),weight_scales=[ratio(v) for v in sw])
        scales[name]=so
    package=dict(parameters=params,residual_parameters=residual,residual_encoding=encoding,
                 checkpoint_sha256=file_hash(source),contract_file_sha256=file_hash(Path(__file__).parents[1]/'model_b_contract.json'),
                 calibrated_on=calibration,step=step,training_binding=binding)
    corrections=[]
    for iteration in range(2):
        integer=IntegerB1(package);sums=torch.zeros(3,device='cuda',dtype=torch.float64);counts=torch.zeros_like(sums)
        for path in calibration:
            rgb=read_image(path);wanted=net(as_tensor(rgb)[None].cuda())*128
            q=integer(torch.tensor(rgb.transpose(2,0,1)[None].astype(np.float32)-128,device='cuda'))
            valid=(wanted.abs()<110)&(q.abs()<125)
            sums+=((q-wanted)*valid).sum((0,2,3),dtype=torch.float64);counts+=valid.sum((0,2,3))
        error=(sums/counts.clamp_min(1)).cpu().numpy();head=params['rgb_head'];before=list(head['bias'])
        for c in range(3):head['bias'][c]-=int(quant(error[c]*131072/head['multipliers'][c]))
        corrections.append(dict(iteration=iteration,mean_rgb255=error.tolist(),before=before,after=list(head['bias'])))
        print('BIAS',iteration,error.tolist(),flush=True)
    package['bias_correction']=corrections
    write_package(package,net,folded,data,out,step,contract,plan)


@torch.inference_mode()
def write_package(package,net,folded,data,out,step,contract,plan):
    params=package['parameters'];residual=package['residual_parameters'];encoding=package['residual_encoding']
    calibration=package['calibrated_on']
    write_json(out/'runtime_parameters.json',package)
    rows={r['id']:r for r in plan['nodes']}
    encoded={name:codec.Parameters(tuple(tuple(v) for v in p['weights']),tuple(p['bias']),tuple(p['multipliers']),rows[name]['activation']=='relu') for name,p in params.items()}
    image=codec.encode(encoded,encoding);padded=image+bytes(20480-len(image))
    assert len(image)==17792 and len(padded)==20480
    (out/'coefficients.bin').write_bytes(image);(out/'coefficients-padded.bin').write_bytes(padded)
    (out/'load_words.hex').write_text(''.join(f'{int.from_bytes(padded[i:i+8],"little"):016x}\n' for i in range(0,20480,8)),encoding='ascii')
    # Preview must start from the deliverable bytes, never the in-memory exporter.
    decoded=codec.runtime_from_image((out/'coefficients-padded.bin').read_bytes());integer=IntegerB1(decoded)
    got,got_residual=codec.decode(padded);assert got==encoded
    for k,r in got_residual.items():assert all(r[n]==encoding[k][n] for n in r)
    rejected=[]
    for name,mutate in [('truncated',lambda b:b[:-1]),('payload_crc',lambda b:b[:1900]+bytes([b[1900]^1])+b[1901:]),
                        ('nonzero_transport_padding',lambda b:b[:-1]+b'\1'),('synthetic_flags',lambda b:b[:36]+b'\3\0\0\0'+b[40:])]:
        try:codec.decode(mutate(padded))
        except ValueError:rejected.append(name)
        else:raise AssertionError(name)
    oracle_params=copy.deepcopy(params)
    for name,p in decoded['parameters'].items():
        oracle_params[name].update(weights=[list(row) for row in p['weights']],bias=list(p['bias']),multipliers=list(p['multipliers']))
    tests=[];rng=np.random.default_rng(20261005)
    samples=[rng.integers(-128,128,(8,12,3),dtype=np.int16),np.full((8,8,3),127,np.int16),np.full((8,8,3),-128,np.int16),read_image(calibration[0])[:12,:16].astype(np.int16)-128]
    for i,x in enumerate(samples):
        ref=execute_integer(contract,x.tolist(),oracle_params,residual)
        values,_=integer(torch.tensor(x.transpose(2,0,1)[None],device='cuda',dtype=torch.float32),True)
        errors={k:int(np.abs(v[0].permute(1,2,0).cpu().numpy()-np.array(ref['tensors'][k])).max()) for k,v in values.items()}
        assert max(errors.values())==0;tests.append(dict(sample=i,shape=list(x.shape),node_errors=errors))
    pictures=Image.new('RGB',(1280,268*len(data['validation'])),'#14202b');draw=ImageDraw.Draw(pictures)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18);checks=[]
    for i,r in enumerate(data['validation']):
        rgb=read_image(r['source']);x=as_tensor(rgb)[None].cuda();fp=net(x);fold=folded(x)
        fold_error=float((fp-fold).abs().max());assert fold_error<1e-4
        y=integer(torch.tensor(rgb.transpose(2,0,1)[None].astype(np.float32)-128,device='cuda'))
        output=(y[0].permute(1,2,0).cpu().numpy()+128).astype(np.uint8)
        Image.fromarray(output).save(out/(r['id']+'_integer720.png'));Image.fromarray(bilinear_2to3(output)).save(out/(r['id']+'_integer1080.png'))
        target=read_image(r['target']);delta=(y-fp*128).abs()
        checks.append(dict(id=r['id'],mae_int8_vs_fp32=float(delta.mean()),max_int8_vs_fp32=float(delta.max()),bn_fold_max=fold_error,
                           mae_target=float(np.abs(output.astype(float)-target).mean()),mae_source=float(np.abs(output.astype(float)-rgb).mean())))
        for j,(a,label) in enumerate([(rgb,'原片'),(target,'确认过的目标'),(output,'现有B1：实际整数包输出')]):
            pictures.paste(Image.fromarray(a).resize((426,240)),(426*j,268*i+28));draw.text((426*j+5,268*i+3),label,font=font,fill='white')
    pictures.save(out/'comparison.jpg',quality=96)
    write_json(out/'verification.json',dict(status='TRAINED_B1_BYTES_RELOADED_INTEGER_ORACLE_PASS',checks=checks,oracle=tests,rejected=rejected,
        bytes=17792,transport_bytes=20480,sha256=file_hash(out/'coefficients-padded.bin'),crc32=zlib.crc32(padded),trained_steps=step,
        graph=plan['model_id'],contract_sha256=package['contract_file_sha256'],integer_ops_only=True,postprocess=False,actual_board=False,
        short_rtl=False,weights_and_residual_roundtrip=True,synthetic_weights=False))
    for path in [Path(__file__),Path(codec.__file__)]:shutil.copy2(path,out/path.name)
    print('TRAINED_PACKAGE',out,checks,flush=True)


def export_qat():
    from .qat_b1_deploy_preview import QuantB1
    cfg,data,binding=setup();contract,plan=contract_guard()
    package=json.loads((OUT/'qat/runtime_000600.json').read_text())
    assert package['qat_binding']['parent']==binding and package['qat_steps']==600
    assert file_hash(OUT/'qat/checkpoint_000600.pt')==package['qat_checkpoint_sha256']
    rows=[json.loads(line) for line in (OUT/'qat/metrics.jsonl').read_text().splitlines()]
    assert [row['step'] for row in rows]==list(range(1,601))
    net=QuantB1(package).cuda().eval();out=OUT/'package-qat600';out.mkdir(exist_ok=False)
    write_package(package,net,net,data,out,2600,contract,plan)
    v=json.loads((out/'verification.json').read_text());v.update(reference='QAT exact integer forward, not unquantized FP32',fp32_steps=2000,qat_steps=600)
    for row in v['checks']:
        row['mae_bytes_vs_qat_forward']=row.pop('mae_int8_vs_fp32')
        row['max_bytes_vs_qat_forward']=row.pop('max_int8_vs_fp32')
        row.pop('bn_fold_max')
    write_json(out/'verification.json',v)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--step',type=int,default=2000);parser.add_argument('--qat',action='store_true');a=parser.parse_args()
    export_qat() if a.qat else main(a.step)
