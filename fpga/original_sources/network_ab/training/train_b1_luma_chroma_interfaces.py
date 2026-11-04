"""Keep15 hard luminance layers; learn continuous region chroma separately.

The exact existing Y contour path is preserved at initialization. Removing
Co/Cg radix carries prevents independently switching hue bits from creating
extra colour rims. Region chroma targets remain stylized, not photographic.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from .train_b1_joint_interfaces import JointInterfaceTraining,driver,preview
from .train_b1_coincident_distance import evaluate as old_evaluate
from .b1_joint_interface_targets import joint_interface_fields
from . import train_b1_minimum_area as m
import numpy as np
from .train_b1_ordered_distance_rims import distance_objective


def chroma(rgb):
    return torch.stack(((rgb[:,0]-rgb[:,2])/2,(2*rgb[:,1]-rgb[:,0]-rgb[:,2])/4),1)


def chroma_objective(p,t,labels):
    delta=chroma(p)-chroma(t);pixel=F.smooth_l1_loss(delta,torch.zeros_like(delta),beta=2/128)
    boundary=torch.zeros_like(labels,dtype=torch.bool)
    boundary[:,:,1:]|=labels[:,:,1:]!=labels[:,:,:-1];boundary[:,1:]|=labels[:,1:]!=labels[:,:-1]
    core=(F.max_pool2d(boundary[:,None].float(),5,1,2)==0)[:,0]
    means=[];flats=[]
    for d,lab,ok in zip(delta,labels,core):
        if not ok.any():continue
        _,idx=torch.unique(lab[ok],return_inverse=True);count=torch.bincount(idx).float();n=len(count);v=d[:,ok]
        mu=d.new_zeros(2,n).scatter_add(1,idx[None].expand_as(v),v)/count
        e=F.smooth_l1_loss(v-mu[:,idx],torch.zeros_like(v),beta=2/128,reduction='none').mean(0)
        per=d.new_zeros(n).scatter_add(0,idx,e)/count;weight=count.sqrt();weight/=weight.sum()
        means.append((F.smooth_l1_loss(mu,torch.zeros_like(mu),beta=2/128,reduction='none').mean(0)*weight).sum())
        flats.append((per*weight).sum())
    mean=torch.stack(means).mean() if means else pixel*0;flat=torch.stack(flats).mean() if flats else pixel*0
    return 8*pixel+4*mean+4*flat,dict(chroma_pixel=pixel,chroma_region_mean=mean,chroma_region_residual_variation=flat)


def luminance_codes(target,labels,rnd):
    y=(target[:,0]+2*target[:,1]+target[:,2])*32+128;result=[]
    for value,lab in zip(y,labels):
        _,idx=torch.unique(lab.flatten(),return_inverse=True);count=torch.bincount(idx).float()
        mu=value.new_zeros(len(count)).scatter_add(0,idx,value.flatten())/count
        code=rnd((mu-16)/16).clamp(0,14).long()
        # Chroma is deliberately neutral in this training-only geometry code;
        # true colours receive their own continuous region supervision.
        result.append((81*code[idx]+40).reshape(lab.shape))
    return torch.stack(result)


class LumaChromaInterfaceTraining(JointInterfaceTraining):
    def configure_distance_decoder(self):
        # The warm decoder already has g11 Y and identity Co/Cg. Never apply
        # the old all-channel ramp: it would destroy continuous chroma.
        pass

    def __init__(self,args,data):
        warm=torch.load(args.warm,map_location='cpu',weights_only=True)
        assert warm.get('radix_luma_continuous_chroma')
        super().__init__(args,data)
        for key,value in self.net.state_dict().items():
            assert torch.equal(value.cpu(),warm['model'][key].cpu()), ('warm initialization changed',key)
        assert self.net.package['residual_encoding']==warm['residual_encoding']
        from .probe_b1_encoder_gauge import load
        _,expected=load(args.warm)
        assert self.net.export()==expected.export()
        print('LUMA_CHROMA_WARM_PASS all_state_equal/all_package_bytes_equal/selective_decoder_preserved',flush=True)

    def batch_style(self,native):
        x,t,specs,flow,valid=super().batch_style(native)
        self.class_labels=luminance_codes(t,self.labels,self.net.round)
        return x,t,specs,flow,valid

    def additional_loss(self,p,t,native):
        truth=self.distance_cache(self.class_labels,8)[:,:14];self.supervision_distance_truth=truth
        field=self.nodes['up1_low_dw'][:,:14];labels=self.labels
        if not native:field=field[...,10:-10,10:-10];truth=truth[...,10:-10,10:-10];labels=labels[...,40:-40,40:-40]
        full,band=distance_objective(field,truth);rank=(F.relu(torch.diff(field,dim=1)).amax(1)/127).square().mean()
        colour,v=chroma_objective(p,t,labels)
        v.update(ordered_distance_full=full,ordered_distance_band=band,luma_rank_violation=rank,motion_aligned_target_fraction=self.motion_fraction)
        return self.args.classification*(8*full+4*band)+10*rank+colour,v

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(radix_luma_continuous_chroma=True,luminance_layers=15,
            ordinal_target='14Y fields from shared nearest luminance interface; no independent chroma thresholds',
            luminance_label='nearest of16+16*k,k0..14 to original stylized region mean Y',
            chroma_target='original stylized Co/Cg, pixel+region residual; no photographic target',
            hard_chroma_decisions_removed=True,constant_palette_requires_runtime=True,extra_inference_operations=0,visual_acceptance=False)
        s.update(rgb_training_target='original stylized Co/Cg pixel and region errors; no full RGB reconstruction loss',
            rgb_objective_weight=None,decoder_ramp_gain={'luma':11,'chroma':'identity'},
            exact_warm_state_and_bytes_asserted=True)
        torch.save(s,path)


@torch.inference_mode()
def evaluate(net,rows):
    result=old_evaluate(net,rows);device=next(net.parameters()).device
    # The inherited full30-field metrics describe the historical target,
    # not this run's objective. Keep them explicitly marked as diagnostics.
    for record,row in zip(result['records'],rows):
        for key in ('ordered_distance_full','ordered_distance_band'):
            if key in record:record['historical_'+key]=record.pop(key)
        x=m.t.r.tensor(m.t.r.read(row['path']))[None]
        target=m.t.r.tensor(m.t.r.read(row['target']))[None]
        labels=torch.from_numpy(np.load(row['region'])['labels'].astype(np.int64))[None].to(device)
        code=luminance_codes(target,labels,net.round)
        field=net(x,all_nodes=True)['up1_low_dw'][:,:14]
        full,band=distance_objective(field,joint_interface_fields(code)[:,:14])
        record.update(luminance_interface_full=float(full),luminance_interface_band=float(band))
    for key in ('ordered_distance_full','ordered_distance_band'):
        if key in result['mean']:result['mean']['historical_'+key]=result['mean'].pop(key)
    for key in ('luminance_interface_full','luminance_interface_band'):
        result['mean'][key]=sum(r[key] for r in result['records'])/len(rows)
    return result


def regression():
    from .qat_b1_deploy_preview import Round
    label=torch.zeros(1,32,48,dtype=torch.long);label[:,:,24:]=1;t=torch.zeros(1,3,32,48);t[:,:, :,24:]=.5
    value,terms=chroma_objective(t,t,label);assert value==0
    wrong=t.clone();wrong[:,1]+= .125;wrong.requires_grad_();loss,_=chroma_objective(wrong,t,label);loss.backward()
    assert loss>0 and wrong.grad[:,1].sum()>0 and torch.isfinite(wrong.grad).all()
    # Common RGB/luma change is not penalized by the chroma-specific term.
    assert chroma_objective(t+.25,t,label)[0]==0
    code=luminance_codes(t,label,Round());assert torch.equal(code//81,torch.where(label==0,7,11))
    print('LUMA_CHROMA_PASS exact_target0/chroma_direction/luma_invariance/15level_label',flush=True)


if __name__=='__main__':
    torch.set_num_threads(2);regression()
    driver.main(training_factory=LumaChromaInterfaceTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
