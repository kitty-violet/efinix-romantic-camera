"""Compile ordinal colour coordinates into the existing binary field decoder.

Thirty tied hinges occupy existing res3 PW outputs. An existing32-to16 PW
encodes their differences as15 region scores. Independent remainder decisions
are thereby replaced with one common coordinate per colour axis. Same B1 graph.
"""
import argparse,copy,math
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from .probe_b1_encoder_gauge import load
from .probe_b1_binary_confidence import set_gain
from .probe_b1_decoder_coverage import fragment
from .build_b1_thirty_binary_regions import identity,code_bits,GROUPS30
from .b1_binary_layout import radix_tables,fields_for_layout
from .train_b1_ternary_regions import region_codes
from .train_b1_joint_region_colour import preview,evaluate
from .runtime import configure_workspace,write_json,file_hash
from . import train_b1_minimum_area as m


@torch.no_grad()
def install_decoder(net,setup):
    assert setup.get('code_layout')=='radix_thermometer_654'
    net.package['residual_encoding']['res3_add']=dict(main_scaled=False,multiplier=1,post_multiplier=131071)
    for key in ('res3_conv2__dw','res3_conv2__pw','up1_low_dw'):identity(net,key)
    key='up1_low_pw';net.weight[key].zero_();net.bias[key].zero_();net.bias[key][:15]=16
    mult=99079;getattr(net,key+'_m').fill_(mult);offset=0
    for table,(a,b) in zip(radix_tables(net.weight[key].device),GROUPS30):
        diff=torch.diff(table,dim=0).T
        net.weight[key][offset:offset+len(diff),a:b,0,0]=diff*mult/131072
        offset+=len(diff)
    set_gain(net,8)


@torch.no_grad()
def install_head(net,coefficient,bias):
    key='res3_conv1__pw';net.weight[key].zero_();net.bias[key].zero_()
    for axis,(a,b) in enumerate(GROUPS30):
        w=127*coefficient[axis];quantum=2**math.ceil(math.log2(max(float(w.abs().max())/120,1/131072)))
        assert quantum<=.5,'Head exceeds existing multiplier range'
        getattr(net,key+'_m')[:,a:b]=quantum*131072
        net.weight[key][a:b,:,0,0]=w[None];net.bias[key][a:b]=127*(bias[axis]-torch.arange(b-a,device=w.device))
    getattr(net,key+'_m')[:,30:]=131071


@torch.inference_mode()
def regression(net,setup):
    level=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).to(net.weight['rgb_head'].device)
    code=(level[:,0]*81+level[:,1]*9+level[:,2])[:,None,None]
    hinge=F.pad(code_bits(code)*127,(0,0,0,0,0,2))
    nodes=fragment(net,hinge,'res3_conv1.pw')
    wanted=fields_for_layout(code,setup)*96+16
    assert torch.equal(nodes['up1_low_pw'][:,:15],wanted)
    palette=(level.float()@level.new_tensor(setup['matrix']).float().T+level.new_tensor(setup['bias'])).clamp(-128,127)
    assert torch.equal(nodes['rgb_head'][:,:,0,0],palette)
    # Uniform fractional coordinates must follow a single adjacent-code path;
    # no independently selected remainder/main combinations are possible here.
    values=torch.linspace(0,14,141,device=hinge.device);h=(values[:,None]-torch.arange(14,device=hinge.device)[None]).clamp(0,1)*127
    quant=net.round(h);assert ((quant.sum(1)/127-values).abs()<1/127).all()
    raw=net.export();assert len(raw)==20480
    actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device=str(hinge.device))
    x=torch.randint(-128,128,(1,3,64,80),device=hinge.device).float();full=net(x/128,all_nodes=True);other,_=actual(x,True)
    assert all(torch.equal(v,other[k]) for k,v in full.items())
    print('COMMON_HINGE_REGRESSION_PASS 1215codes/score_fields/coordinate_sum/integer_nodes/20KiB',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    configure_workspace();torch.set_num_threads(2);torch.manual_seed(819);dest=Path(a.out);dest.mkdir(exist_ok=False)
    state,net=load(a.checkpoint);net=net.cuda();setup=copy.deepcopy(state['binary_coverage720']);data=m.load_data(a.data)
    levels=torch.cartesian_prod(torch.arange(15,device='cuda'),torch.arange(9,device='cuda'),torch.arange(9,device='cuda')).float()
    palette=(levels@levels.new_tensor(setup['matrix']).T+levels.new_tensor(setup['bias'])+128).clamp(0,255);pk=tuple(map(tuple,palette.cpu().tolist()))
    grouped={}
    for row in data['training']:grouped.setdefault(row['video'],[]).append(row)
    chosen=[]
    for group in grouped.values():
        group=sorted(group,key=lambda r:r['id']);chosen.extend(group[i] for i in sorted(set((0,len(group)//2,len(group)-1))))
    xx=[];yy=[]
    with torch.no_grad():
        for row in chosen:
            nodes=net(m.t.r.tensor(m.t.r.read(row['path']))[None],all_nodes=True)
            feature=nodes['res3_conv1.dw'][0,:,::3,::3].flatten(1).T
            code=torch.from_numpy(region_codes(row['target'],row['region'],pk).astype(np.int64))[None].cuda()
            coordinate=torch.stack((code//81,code//9%9,code%9),1).float()
            target=F.avg_pool2d(coordinate,4)[0,:,::3,::3].flatten(1).T;xx.append(feature);yy.append(target)
        x=torch.cat(xx);target=torch.cat(yy);mu=x.mean(0);sd=x.std(0).clamp_min(1)
        design=torch.cat(((x-mu)/sd,torch.ones(len(x),1,device='cuda')),1);penalty=torch.eye(33,device='cuda')*.01;penalty[-1,-1]=0
        theta=torch.linalg.solve(design.T@design/len(x)+penalty,design.T@target/len(x));w=theta[:-1].T/sd[None];b=theta[-1]-w@mu
        fit_rmse=float(((design@theta-target)**2).mean().sqrt());install_head(net,w,b);install_decoder(net,setup);regression(net,setup)
        setup.update(gain=8,source_readout_not_fitted=False,common_coordinate_hinges=True)
        raw=net.export();actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cuda')
        for row in data['validation']:
            x=m.t.r.tensor(m.t.r.read(row['path']))[None];nodes=net(x,all_nodes=True);other,_=actual(x*128,True)
            assert all(torch.equal(v,other[k]) for k,v in nodes.items())
        saved=copy.deepcopy(state);saved.update(model={k:v.cpu() for k,v in net.state_dict().items()},binary_coverage720=setup,
            residual_encoding=copy.deepcopy(net.package['residual_encoding']),common_coordinate_hinges=True,
            source_readout_not_fitted=False,source_readout_train_ids=[r['id'] for r in chosen],step=0,visual_acceptance=False)
        torch.save(saved,dest/'initial.pt');preview(net,data['validation'],dest/'preview');review=evaluate(net,data['validation'])
    write_json(dest/'review.json',dict(source_prediction=True,readout_fit_rmse=fit_rmse,training_ids=[r['id'] for r in chosen],
        all_six720_node_error=0,byte_size=20480,validation=review,visual_acceptance=False,parent_sha256=file_hash(a.checkpoint)))
    print('COMMON_HINGE_SOURCE_SEED_COMPLETE',fit_rmse,review['mean'],flush=True)


if __name__=='__main__':main()
