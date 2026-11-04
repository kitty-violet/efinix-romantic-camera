"""TRAIN-only least-squares calibration of the existing shared RGB matrix.

The linear fit is only an initializer: actual integer output, clipping, rounding
and all nodes are checked afterwards. Geometry weights and fixed operators stay
unchanged. No LUT, postfilter, new parameter slot or validation fitting.
"""
import argparse
import copy
import time
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image, ImageDraw, ImageFont
from . import train_b1_minimum_area as m
from .build_b1_binary_palette import calibrate_rows
from .train_b1_joint_region_colour import joint_region_loss
from .runtime import configure_workspace, contract_guard, write_json, file_hash
from .engine import set_reproducible


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',default='joint-rgb-calibration');args=ap.parse_args()
    configure_workspace(); contract_guard(); set_reproducible(20261006283)
    dest=m.OUT/args.name;dest.mkdir(exist_ok=False)
    path=m.OUT/'joint-field-colour-qat/candidate_000500.pt'
    data=m.load_data(m.OUT/'hierarchical-targets-v2/data.json')
    state=torch.load(path,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin').cuda()
    net.load_state_dict(state['model']);net.package['residual_encoding']=copy.deepcopy(state['residual_encoding']);net.quantized=True
    initial=copy.deepcopy(net.state_dict());gram=torch.zeros(16,16,device='cuda',dtype=torch.float64)
    cross=torch.zeros(16,3,device='cuda',dtype=torch.float64);samples=0;began=time.perf_counter()
    count_by_source={}
    for row in data['training']:count_by_source[row['video']]=count_by_source.get(row['video'],0)+1
    records=[]
    # First3 finalDW kernels are identical, positive and exactly DC1.
    kernel=net.weight['up2_high_dw'][0:1].clone()
    assert torch.equal(kernel.expand(3,-1,-1,-1),net.weight['up2_high_dw'][:3])
    assert float(kernel.sum())==1
    for i,row in enumerate(data['training']):
        x=m.t.r.tensor(m.t.r.read(row['path']))[None]
        target=m.t.r.tensor(m.t.r.read(row['target']))[None]
        field=net(x,all_nodes=True)['up2_low_dw'][:,:15]/127
        # Linearized features before output requantization. The exact deployed
        # result is separately replayed below, not equated with this surrogate.
        field=field.repeat_interleave(2,-2).repeat_interleave(2,-1)
        field=F.conv2d(F.pad(field,(1,1,1,1),mode='replicate'),kernel.expand(15,-1,-1,-1),groups=15)
        oy,ox=(i//8)%8,i%8
        f=field[0,:,oy::8,ox::8].flatten(1).T
        f=torch.cat((f,torch.ones(len(f),1,device=f.device)),1).double()
        truth=((target[0,:,oy::8,ox::8]+1)*64).flatten(1).T.double()
        lab=m.load_region(row['region']);sizes=np.bincount(lab.ravel())
        weight=torch.tensor(1/np.sqrt(sizes[lab[oy::8,ox::8]].ravel()),device='cuda',dtype=torch.float64)
        weight=weight/(weight.sum()*count_by_source[row['video']])
        gram+=f.T@(f*weight[:,None]);cross+=f.T@(truth*weight[:,None]);samples+=len(f)
        records.append(row['id'])
        if (i+1)%25==0:print('RGB_CALIBRATION_DATA',i+1,len(data['training']),flush=True)
    # Small ridge only stabilizes collinear bit fields. Anchor to the current
    # matrix so a poorly observed field cannot acquire a large arbitrary gain.
    prior=torch.cat((net.weight['up2_low_pw'][:3,:15,0,0].T*127,net.bias['up2_low_pw'][:3][None]),0).double()
    lam=1e-4*float(gram.diag().mean());identity=torch.eye(16,device='cuda',dtype=torch.float64)
    solution=torch.linalg.solve(gram+lam*identity,cross+lam*prior)
    net.weight['up2_low_pw'][:3,:15,0,0]=solution[:15].T.float()/127
    net.bias['up2_low_pw'][:3]=solution[15].float()
    calibrate_rows(net,['up2_low_pw'])
    calibrated=copy.deepcopy(net.state_dict())
    # Only the declared layer's coefficients/multipliers may differ.
    changed=[key for key in initial if not torch.equal(initial[key],calibrated[key])]
    assert changed and all('up2_low_pw' in key for key in changed)
    images={};evaluations=[];by={r['id']:r for r in data['validation']}
    for name,model in (('before',initial),('calibrated',calibrated)):
        net.load_state_dict(model);raw=net.export();actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cuda')
        sub=dest/name;sub.mkdir();rows=[];pics={}
        for row in data['validation']:
            x=m.t.r.tensor(m.t.r.read(row['path']))[None];target=m.t.r.tensor(m.t.r.read(row['target']))[None]
            node=net(x,all_nodes=True);replay,_=actual(x*128,True)
            error=max(float((v-replay[key]).abs().max()) for key,v in node.items() if key!='input');assert error==0
            p=replay['rgb_head']/128;lab=torch.tensor(m.load_region(row['region']).astype(np.int64),device='cuda')[None]
            loss,terms=joint_region_loss(p,target,lab)
            delta=(p-target)*128;r,g,b=delta.unbind(1)
            rows.append(dict(id=row['id'],all_node_error=error,**m.t.r.metrics(p,target),
                             **{k:float(v) for k,v in terms.items()},bias_Co=float(((r-b)/2).mean()),bias_Cg=float((2*g-r-b).mean()/4)))
            im=m.t.r.c.rgb_image(p[0]);im.save(sub/(row['id']+'.png'));pics[row['id']]=im
        (sub/'coefficients-padded.bin').write_bytes(raw)
        saved=dict(state,model=copy.deepcopy(model),rgb_least_squares_calibration=True,
                   calibration_training_ids=records,parent_checkpoint=str(path),visual_acceptance=False,
                   calibration_optimizer_state_valid=False)
        # The parent's optimizer/RNG state is not a resume point for a calibrated model.
        for key in ('optimizer','rng','torch_rng','cuda_rng'):saved.pop(key,None)
        torch.save(saved,sub/'candidate.pt')
        evaluations.append(dict(variant=name,records=rows,bytes=len(raw)))
        images[name]=pics
        print('RGB_CALIBRATION_EVAL',name,rows,flush=True)
    board=Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        imgs=[Image.open(by[key]['path']),Image.open(by[key]['target']),images['before'][key],images['calibrated'][key]]
        for j,(im,title) in enumerate(zip(imgs,('原图','修订目标','校准前·真实整数','只校准颜色·真实整数'))):
            board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+4,314*i+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'comparison.png')
    write_json(dest/'results.json',dict(training_ids=records,samples=samples,seconds=time.perf_counter()-began,
        changed_state_keys=changed,ridge=lam,records=evaluations,parent_sha256=file_hash(path),
        code_sha256=file_hash(__file__),training_only_fit=True,geometry_weights_unchanged=True,
        exact_integer_replay=True,inference_added_operators=0,visual_acceptance=False,temporal_acceptance=False))
    print('RGB_CALIBRATION_COMPLETE',str(dest),flush=True)


if __name__=='__main__':main()
