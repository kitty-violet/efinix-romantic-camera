"""Calibrate final integer bias on TRAIN sources, without extra inference ops."""
import argparse,copy,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .probe_b1_cartoon_int8 import IntegerB1,RUN
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible
from .publish_b1_cartoon_budget import load
from .train_b1_distillation import as_tensor,read_image
from network_ab.contract_reference import execute_integer


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--step',type=int,default=2000);a=ap.parse_args()
    root=configure_workspace();set_reproducible(2026100407);contract,_=contract_guard();run=root/'runs'/RUN
    raw=run/f'int8-probe-{a.step}';out=run/f'int8-bias-corrected-{a.step}';out.mkdir(exist_ok=False)
    package=json.loads((raw/'runtime_parameters.json').read_text());model=load(run/f'final-{a.step}/b1_cartoon_fp32.pt')
    manifest=json.loads((run/'data_manifest.json').read_text(encoding='utf-8'));train={r['current'] for r in manifest['pairs']};assert set(package['calibrated_on'])<=train
    corrections=[]
    for iteration in range(2):
        net=IntegerB1(package);sums=torch.zeros(3,device='cuda',dtype=torch.float64);counts=torch.zeros_like(sums)
        with torch.inference_mode():
            for path in package['calibrated_on']:
                rgb=read_image(path);x=as_tensor(rgb)[None].cuda();wanted=model(x)*128
                q=net(torch.tensor(rgb.transpose(2,0,1)[None].astype(np.float32)-128,device='cuda'))
                valid=(wanted.abs()<110)&(q.abs()<125)
                sums+=((q-wanted)*valid).sum((0,2,3),dtype=torch.float64);counts+=valid.sum((0,2,3))
        mean=(sums/counts).cpu().numpy();p=package['parameters']['rgb_head'];before=list(p['bias'])
        for c in range(3):
            delta=float(mean[c])*131072/p['multipliers'][c];p['bias'][c]-=int(np.sign(delta)*np.floor(abs(delta)+.5));assert abs(p['bias'][c])<2**31
        corrections.append(dict(iteration=iteration,train_mean_signed_rgb255=mean.tolist(),old_bias=before,new_bias=list(p['bias'])))
        print('INT8_FINAL_BIAS',iteration,mean.tolist(),flush=True)
    package['final_bias_correction']=corrections;package['uncorrected_parameters_sha256']=file_hash(raw/'runtime_parameters.json')
    write_json(out/'runtime_parameters.json',package);net=IntegerB1(package)
    rng=np.random.default_rng(20261004);tests=[]
    for index,image in enumerate([rng.integers(-128,128,(8,12,3),dtype=np.int16),np.full((8,8,3),127,dtype=np.int16),np.full((8,8,3),-128,dtype=np.int16)]):
        ref=execute_integer(contract,image.tolist(),package['parameters'],package['residual_parameters'])
        values,_=net(torch.tensor(image.transpose(2,0,1)[None],device='cuda',dtype=torch.float32),True)
        errors={k:int(np.abs(v[0].permute(1,2,0).cpu().numpy()-np.array(ref['tensors'][k])).max()) for k,v in values.items()};assert max(errors.values())==0
        tests.append(dict(sample=index,node_errors=errors))
    records=[]
    with torch.inference_mode():
        for row in manifest['validation']:
            rgb=read_image(row['source']);fp=model(as_tensor(rgb)[None].cuda())*128
            y=net(torch.tensor(rgb.transpose(2,0,1)[None].astype(np.float32)-128,device='cuda'))
            image=(y[0].permute(1,2,0).cpu().numpy()+128).astype(np.uint8);Image.fromarray(image).save(out/(row['id']+'.png'))
            d=(y-fp).abs();records.append(dict(id=row['id'],mae_to_fp32_rgb255=float(d.mean()),p95_to_fp32_rgb255=float(torch.quantile(d.flatten(),.95)),max_to_fp32_rgb255=float(d.max())))
    write_json(out/'verification.json',dict(status='SMALL_INTEGER_GRAPH_PASS_NOT_RTL',corrections=corrections,records=records,small_graph_tests=tests,
        source='Calibration only on21 training images, no validation fitting',runtime_parameters_sha256=file_hash(out/'runtime_parameters.json'),
        inference_ops_added=False,hardware_release=False,qat=False,fp32_weights_sha256=package['fp32_sha256']))
    print('BIAS_CORRECTION_COMPLETE',[(r['id'],round(r['mae_to_fp32_rgb255'],3)) for r in records],flush=True)


if __name__=='__main__':main()
