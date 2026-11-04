"""Replace redundant linear late features by trainable complementary ReLUs.

This is a TRAIN-calibrated initialization, not an exact reparameterization of
the eight original channels, nor a sharpening postfilter. Existing B1 layers,
activations and package size are unchanged. Initial function error is measured.
"""
import argparse,copy,shutil
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .audit_b1_last_stage_nonlinearity import preactivation
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.no_grad()
def deployed(net,key):
    scale=(getattr(net,key+'_m')/131072).float().flatten()
    return net.round(net.weight[key]/scale[:,None,None,None]).clamp(-127,127)*scale[:,None,None,None],net.round(net.bias[key]/scale)*scale


@torch.no_grad()
def install(net,features,outputs):
    # Three highly correlated pairs were measured, rather than inferred from
    # the old channel labels. Preserve the two independent remaining channels.
    basis=features[:,[0,1,2,6,7]].double();want=outputs.double()
    mu=basis.mean(0);std=basis.std(0).clamp_min(1)
    a=torch.cat(((basis-mu)/std,basis.new_ones(len(basis),1)),1)
    penalty=torch.eye(6,dtype=a.dtype)*len(a)*1e-6;penalty[-1,-1]=0
    coef=torch.linalg.solve(a.T@a+penalty,a.T@want)
    weight=(coef[:5]/std[:,None]).T.float();bias=(coef[-1]-mu@weight.double().T).float()
    thresholds=torch.quantile(features[:,:3],.5,dim=0).round()
    pw,pb=deployed(net,'up2_low_pw');dw,db=deployed(net,'up2_high_dw')
    original_m=net.up2_low_pw_m.clone();dw_m=net.up2_high_dw_m.clone()
    for c in range(3):
        for dest,sign in ((c,1),(c+3,-1)):
            net.weight['up2_low_pw'][dest].copy_(pw[c]);net.bias['up2_low_pw'][dest].copy_(pb[c])
            net.up2_low_pw_m[:,dest].copy_(original_m[:,c])
            net.weight['up2_high_dw'][dest].copy_(sign*dw[c])
            net.bias['up2_high_dw'][dest].copy_(sign*(db[c]-thresholds[c]))
            net.up2_high_dw_m[:,dest].copy_(dw_m[:,c])
    head=net.weight['rgb_head'];head.zero_()
    head[:,:3,0,0].copy_(weight[:,:3]);head[:,3:6,0,0].copy_(-weight[:,:3]);head[:,6:,0,0].copy_(weight[:,3:])
    net.bias['rgb_head'].copy_(bias+weight[:,:3]@thresholds)
    # Existing per-output Q17 metadata; enough range for the fitted RGB rows.
    mult=(head.abs().flatten(1).amax(1)*131072/120).ceil().clamp(1,131071)
    net.rgb_head_m.copy_(mult.double()[None,:,None,None])
    for key,w in net.weight.items():
        scale=(getattr(net,key+'_m')/131072).float().flatten()
        assert net.round(w/scale[:,None,None,None]).abs().max()<=127
    residual=basis@weight.double().T+bias.double()-want
    return dict(thresholds=thresholds.tolist(),calibration_prequantized_compression_mae=float(residual.abs().mean()),
                five_feature_rgb_weights=weight.tolist(),note='Approximate TRAIN least-squares compression before paired activation; no target labels used')


def regression():
    v=torch.arange(128.)
    for threshold in (0,27,64,127):
        assert torch.equal((v-threshold).clamp(0,127)-(threshold-v).clamp(0,127)+threshold,v)
    print('COMPLEMENTARY_BASIS_PASS integer0to127/all_thresholds',flush=True)


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    torch.set_num_threads(2);regression();state,old=load(args.checkpoint);net=copy.deepcopy(old);out=Path(args.out);out.mkdir(exist_ok=False)
    data=m.load_data(args.data);features=[];outputs=[];calibration=[]
    tensor=lambda z:torch.from_numpy(z.transpose(2,0,1).copy())[None].float()/128-1
    hw,hb=deployed(old,'rgb_head')
    for video in sorted({r['video'] for r in data['training']}):
        rows=[r for r in data['training'] if r['video']==video]
        for i in sorted(set((0,len(rows)//2,len(rows)-1))):
            row=rows[i];z=old(tensor(m.t.r.read(row['path'])),all_nodes=True)['up2_high_dw'][0,:,40:-40:8,40:-40:8].flatten(1).T
            features.append(z);outputs.append(z@hw[:,:,0,0].T+hb);calibration.append(row['id'])
    spec=install(net,torch.cat(features),torch.cat(outputs));spec['train_ids']=calibration
    ims={};records=[]
    for row in data['validation']:
        x=tensor(m.t.r.read(row['path']));target=tensor(m.t.r.read(row['target']));before=old(x,all_nodes=True);after=net(x,all_nodes=True)
        rt=m.t.r.IntegerB1(m.t.r.runtime_from_image(net.export()),device='cpu');actual,_=rt(net.round(x*128),True)
        assert all(torch.equal(v,actual[k]) for k,v in after.items()) and len(net.export())==20480
        # The encoder and pre-final spatial decoder are unchanged; only the
        # last colour bank, final DW and RGB head have a new initialization.
        for node in net.plan['nodes']:
            if node['id']=='up2_low_pw':break
            assert torch.equal(before[node['id']],after[node['id']])
        diff=(after['rgb_head']-before['rgb_head']).abs();pre=preactivation(net,'up2_high_dw',after['up2_nn'])
        records.append(dict(id=row['id'],initial_output_change_mae=float(diff.mean()),initial_output_change_p99=float(torch.quantile(diff.flatten(),.99)),
            old_target_mae=float((before['rgb_head']/128-target).abs().mean()*128),new_target_mae=float((after['rgb_head']/128-target).abs().mean()*128),
            below0_fraction=(pre<0).float().mean((0,2,3)).tolist(),above127_fraction=(pre>127).float().mean((0,2,3)).tolist(),all_nodes_error=0))
        ims[row['id']]=[m.t.r.c.rgb_image(target[0]),m.t.r.c.rgb_image(before['rgb_head'][0]/128),m.t.r.c.rgb_image(after['rgb_head'][0]/128)]
        print('COMPLEMENTARY_SEED',records[-1],flush=True)
    board=Image.new('RGB',(1260,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    for i,(rid,box,title) in enumerate(m.t.r.c.CROPS):
        for j,(im,label) in enumerate(zip(ims[rid],('目标','修改前实际B1','互补响应初始化·未训练'))):
            board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+3,314*i+4),title+' | '+label,font=font,fill='white')
    board.save(out/'comparison.png')
    # Warm initialization only. Stale optimizer/RNG must never be mistaken
    # for a valid resume state after changing coefficient coordinates.
    state['model']=copy.deepcopy(net.state_dict());state['step']=0
    for key in ('optimizer','opt'):state.pop(key,None)
    state.update(complementary_final_basis=spec,extra_inference_operations=0,visual_acceptance=False,warm_only=True)
    torch.save(state,out/'initial.pt');(out/'coefficients-padded.bin').write_bytes(net.export());shutil.copy2(__file__,out/Path(__file__).name)
    write_json(out/'review.json',dict(spec=spec,records=records,checkpoint_sha256=file_hash(args.checkpoint),data_sha256=file_hash(args.data),source_sha256=file_hash(__file__),initial_function_is_approximate=True,visual_acceptance=False))


if __name__=='__main__':main()
