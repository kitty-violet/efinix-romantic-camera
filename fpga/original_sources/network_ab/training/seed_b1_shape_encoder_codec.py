"""Align the learned region encoder with the independently tested B1 codec.

The ternary branch supplies nonlinear source features; the painting codec
supplies its unchanged decoder. Only the existing final encoder PW is fitted
on TRAIN images. Target encodings are fitting labels, never runtime inputs.
"""
import argparse,copy
from pathlib import Path
import numpy as np
import torch
from . import train_b1_minimum_area as m
from .train_b1_joint_region_colour import evaluate,preview
from .runtime import configure_workspace,contract_guard,file_hash,write_json


def load(path):
    state=torch.load(path,map_location='cpu',weights_only=True)
    net=m.t.r.initialize();net.load_state_dict(state['model']);net.quantized=True
    net.package['residual_encoding']=copy.deepcopy(state['residual_encoding'])
    return state,net


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--encoder',required=True);ap.add_argument('--codec',required=True)
    ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    configure_workspace();contract_guard();torch.set_num_threads(2)
    dest=Path(args.out);dest.mkdir(exist_ok=False);data=m.load_data(args.data)
    state,net=load(args.encoder);codec_state,teacher=load(args.codec)
    assert codec_state.get('painting_codec');groups={}
    for row in data['training']:groups.setdefault(row['video'],[]).append(row)
    rows=[]
    for group in groups.values():rows.extend(group[i] for i in sorted(set(np.linspace(0,len(group)-1,8).astype(int))))
    xx=torch.zeros(33,33,dtype=torch.float64,device='cuda');xy=torch.zeros(33,32,dtype=torch.float64,device='cuda')
    target_energy=torch.zeros(32,dtype=torch.float64,device='cuda');total=0
    for row in rows:
        source=m.t.r.tensor(m.t.r.read(row['path']))[None];target=m.t.r.tensor(m.t.r.read(row['target']))[None]
        feature=net(source,all_nodes=True)['res3_conv2.dw'][0,:,8:-8:3,8:-8:3].flatten(1).T.double()
        wanted=teacher(target,all_nodes=True)['res3_add'][0,:,8:-8:3,8:-8:3].flatten(1).T.double()
        feature=torch.cat((feature,torch.ones(len(feature),1,device='cuda',dtype=torch.float64)),1)
        xx+=feature.T@feature/len(feature);xy+=feature.T@wanted/len(feature)
        target_energy+=wanted.square().mean(0);total+=1
    xx/=total;xy/=total;target_energy/=total
    # Standardize the input covariance so differently-scaled channels do not
    # acquire an arbitrary ridge penalty. Intercept is not regularized.
    mean=xx[:32,32];var=(xx.diag()[:32]-mean.square()).clamp_min(1)
    cov=xx[:32,:32]-mean[:,None]*mean[None]
    cross=xy[:32]-mean[:,None]*xy[32]
    eye=torch.eye(32,device='cuda',dtype=torch.float64)
    w=torch.linalg.solve(cov/var.sqrt()[:,None]/var.sqrt()[None]+.001*eye,cross/var.sqrt()[:,None])/var.sqrt()[:,None]
    b=xy[32]-mean@w
    original=copy.deepcopy(net.state_dict());teacher_dict=teacher.state_dict();new=copy.deepcopy(original)
    for key in new:
        if any(part in key for part in ('up1_','up2_','rgb_head')):new[key]=teacher_dict[key]
    net.load_state_dict(new)
    key='res3_conv2__pw';net.weight[key].copy_(w.T.float()[:,:,None,None]);net.bias[key].copy_(b.float())
    mm=(net.weight[key].abs().flatten(1).amax(1).clamp_min(.001)*131072/112).ceil().clamp(1,131071)
    getattr(net,key+'_m').copy_(mm.double()[None,:,None,None])
    net.package['residual_encoding']['res3_add']=dict(main_scaled=False,multiplier=1,post_multiplier=131071)
    fit_error=(target_energy-2*(torch.cat((w,b[None]),0)*xy).sum(0)+
               (torch.cat((w,b[None]),0)*(xx@torch.cat((w,b[None]),0))).sum(0)).clamp_min(0).sqrt()
    saved=dict(model=copy.deepcopy(net.state_dict()),residual_encoding=copy.deepcopy(net.package['residual_encoding']),
        contract_sha256=state['contract_sha256'],qat=True,step=0,shape_encoder_codec=True,
        encoder_checkpoint=args.encoder,encoder_sha256=file_hash(args.encoder),codec_checkpoint=args.codec,
        codec_sha256=file_hash(args.codec),fit_ids=[r['id'] for r in rows],train_only=True,
        manual_affine_readout_fit=True,extra_inference_operations=0,visual_acceptance=False)
    torch.save(saved,dest/'initial.pt');raw=net.export();(dest/'coefficients-padded.bin').write_bytes(raw)
    actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cuda');errors=[]
    for row in data['validation']:
        x=m.t.r.tensor(m.t.r.read(row['path']))[None];a=net(x,all_nodes=True);b,_=actual(x*128,True)
        errors.append(max(float((v-b[k]).abs().max()) for k,v in a.items()))
    assert max(errors)==0 and len(raw)==20480
    preview(net,data['validation'],dest/'preview');review=evaluate(net,data['validation'])
    write_json(dest/'review.json',dict(args=vars(args),fitted_rows=len(rows),fit_ids=saved['fit_ids'],
        unquantized_readout_train_rmse=fit_error.tolist(),validation=review,all_node_errors=errors,
        bytes=len(raw),source_prediction=True,visual_acceptance=False))
    print('SHAPE_ENCODER_CODEC_FIT',len(rows),float(fit_error.mean()),review['mean'],flush=True)


if __name__=='__main__':main()
