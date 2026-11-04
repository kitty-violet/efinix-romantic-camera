"""Use B1's existing512 readout weights to learn14 coherent Y predicates.

The fixed identity readout previously required32 preceding feature channels
to be predefined distance/chroma fields. Supervise the14 actual readout Y
channels instead, and let its32 inputs be latent features. The spatial tail,
g11 decisions, continuous chroma and all inference operators stay unchanged.
"""
import argparse,copy,sys
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from .train_b1_tight_luma_rims import TightLumaRimTraining,driver,preview
from .train_b1_reference_profile_distance import evaluate as base_evaluate
from .train_b1_radix_reachable_ordinal_band import OrdinalBandB1
from .train_b1_winner_surrogate import ExactValueProxyGradient
from .train_b1_luma_chroma_interfaces import luminance_codes,chroma_objective
from .train_b1_direct_luma_interfaces import luma_decision_loss
from .train_b1_ordered_distance_rims import distance_objective
from .b1_joint_interface_targets import joint_interface_fields
from .audit_b1_rim_tolerance import rim_terms
from .centered_b1_optimizer import CenteredB1Adam,calibration_means
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .build_b1_thirty_binary_regions import code_bits
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


class ReadoutB1(OrdinalBandB1):
    def training_activation_proxy(self,name,value,pre,r):
        if name=='up1_low_pw':
            clipped=pre.clamp(0,127)
            return ExactValueProxyGradient.apply(value,clipped+.05*(pre-clipped))
        return super().training_activation_proxy(name,value,pre,r)


def backend_palette(net):
    code=torch.arange(1215,device=next(net.parameters()).device)
    z=torch.cat((code_bits(code[:,None,None])[:,:14]*127,
                 (code//9%9*8)[:,None,None,None],(code%9*8)[:,None,None,None]),1).float()
    return fragment(net,z,'up1_low_pw')['rgb_head'][:,:,0,0]


class TrainableLumaReadoutTraining(TightLumaRimTraining):
    def __init__(self,args,data):
        super().__init__(args,data)
        assert torch.all(self.net.up1_low_pw_m==2048),'Use the verified exact fine-scale warm checkpoint'
        self.net.__class__=ReadoutB1
        key='up1_low_pw';self.fixed.pop(key)
        self.net.weight[key].requires_grad_(True);self.net.bias[key].requires_grad_(True)
        # Rebuild the centered optimizer from the identical TRAIN calibration.
        # The warm run's moments are deliberately not presented as exact resume.
        groups=[dict(params=list(g['params']),lr=g['lr'],**({'row_gain':g['row_gain']} if 'row_gain' in g else {})) for g in self.opt.param_groups]
        scale=(self.net.up1_low_pw_m/131072).float().flatten()
        groups.extend([dict(params=[self.net.weight[key]],lr=args.lr,row_gain=(scale*256)[:,None,None,None]),
                       dict(params=[self.net.bias[key]],lr=.008)])
        means={g['node_key']:g['input_mean'] for g in self.opt.param_groups if 'node_key' in g}
        mu=[]
        with torch.no_grad():
            for rid in self.centering_calibration_ids:
                row=self.by_id[rid];x=m.t.r.tensor(m.t.r.read(row['path']))[None]
                mu.append(calibration_means(self.net,x)[key])
            means[key]=torch.stack(mu).mean(0)
            self.palette=backend_palette(self.net)/128
            self.pk=tuple(map(tuple,(self.palette.cpu().numpy()*128+128).tolist()))
        self.opt=CenteredB1Adam(groups,self.net,means);self.base_lrs=[g['lr'] for g in self.opt.param_groups]

    def additional_loss(self,p,t,native):
        truth=self.distance_cache(self.class_labels,8)[:,:14]
        self.supervision_distance_truth=truth
        with torch.no_grad():
            # Supervision now enters AFTER the trainable readout. Running it
            # through that readout again would create a moving/wrong target.
            want=fragment(self.net,F.pad(self.net.round(truth),(0,0,0,0,0,2)),'up1_low_pw')['up1_high_dw'][:,:14]
        field=self.nodes['up1_low_pw'][:,:14];pred=self.nodes['up1_high_dw'][:,:14];labels=self.labels
        if not native:
            field=field[...,10:-10,10:-10];truth=truth[...,10:-10,10:-10]
            pred=pred[...,20:-20,20:-20];want=want[...,20:-20,20:-20];labels=labels[...,40:-40,40:-40]
        full,band=distance_objective(field,truth);kl,near=luma_decision_loss(pred,want)
        rank=(F.relu(torch.diff(field,dim=1)).amax(1)/127).square().mean()
        colour,v=chroma_objective(p,t,labels);rim,_=rim_terms(p,t,labels,2)
        v.update(ordered_distance_full=full,ordered_distance_band=band,luminance_decision_kl=kl,luminance_decision_band_kl=near,
                 luma_rank_violation=rank,two_pixel_region_centred_bright_shoulder=rim,motion_aligned_target_fraction=self.motion_fraction)
        return self.args.classification*(2*full+band+8*kl+4*near)+10*rank+colour+rim,v

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(trainable_luma_readout=True,luma_supervision_node='up1_low_pw',encoder32_channels_are_latent=True,
                 ordinal_target='14 shared-interface Y fields AFTER existing trainable up1_low_pw',
                 backend_palette_node='up1_low_pw',fixed_readout_identity_removed=True,
                 extra_inference_operations=0,visual_acceptance=False)
        torch.save(s,path)

    def preflight_extras(self):
        result=super().preflight_extras()
        w=self.net.weight['up1_low_pw'];g=w.grad
        assert g is not None and torch.isfinite(g).all() and g.norm()>0
        assert self.opt.state[w]['exp_avg'].abs().max()>0
        result['new_readout_gradient_norm']=float(g.norm())
        result['new_readout_optimizer_active']=True
        return result


@torch.inference_mode()
def evaluate(net,rows):
    result=base_evaluate(net,rows)
    for rec,row in zip(result['records'],rows):
        x=m.t.r.tensor(m.t.r.read(row['path']))[None];t=m.t.r.tensor(m.t.r.read(row['target']))[None]
        labels=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None].to(x.device)
        truth=joint_interface_fields(luminance_codes(t,labels,net.round))[:,:14]
        nodes=net(x,all_nodes=True);full,band=distance_objective(nodes['up1_low_pw'][:,:14],truth)
        rim,_=rim_terms(nodes['rgb_head']/128,t,labels,2)
        rec.update(luminance_interface_full=float(full),luminance_interface_band=float(band),two_pixel_region_centred_bright_shoulder=float(rim))
    for key in ('luminance_interface_full','luminance_interface_band','two_pixel_region_centred_bright_shoulder'):
        result['mean'][key]=sum(r[key] for r in result['records'])/len(rows)
    return result


@torch.no_grad()
def prepare(a):
    state,base=load(a.checkpoint);assert state.get('direct_luminance_planes')
    net=copy.deepcopy(base);net.up1_low_pw_m.fill_(2048)
    # Both original Y identity and C mean weights are exactly representable.
    scale=(net.up1_low_pw_m/131072).float().flatten()[:,None,None,None]
    assert torch.equal(net.weight['up1_low_pw'],net.round(net.weight['up1_low_pw']/scale)*scale)
    assert (net.weight['up1_low_pw']/scale).abs().max()<=127
    out=Path(a.out);out.mkdir(exist_ok=False);raw=net.export();assert len(raw)==20480
    runtime=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')
    for row in m.load_data(a.data)['validation']:
        x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
        aa=base(x,all_nodes=True);bb=net(x,all_nodes=True);cc,_=runtime(net.round(x*128),True)
        assert all(torch.equal(aa[k],bb[k]) and torch.equal(bb[k],cc[k]) for k in aa)
        print('READOUT_FINE_SCALE',row['id'],'all_nodes0',flush=True)
    lev=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float()
    expected=(lev@torch.tensor(state['binary_coverage720']['matrix']).T+torch.tensor(state['binary_coverage720']['bias'])).clamp(-128,127)
    assert torch.equal(backend_palette(net),expected)
    saved={k:v for k,v in state.items() if k not in ('optimizer','rng','torch_rng','cuda_rng')}
    saved.update(model=copy.deepcopy(net.state_dict()),readout_fine_scale_prepared=True,readout_fine_scale_all_nodes_error=0)
    torch.save(saved,out/'initial.pt');write_json(out/'review.json',dict(all_nodes_error=0,actual_byte_nodes_error=0,
        backend_constant1215_error=0,bytes=20480,new_operators=0,visual_acceptance=False,checkpoint_sha256=file_hash(a.checkpoint),source_sha256=file_hash(__file__)))
    # A separate differentiable pass verifies new readout gradients and that
    # adding the derivative proxy never alters actual integer values/bytes.
    with torch.enable_grad():
        x=x[...,:128,:160];before=net(x,all_nodes=True);raw=net.export();net.__class__=ReadoutB1;after=net(x,all_nodes=True)
        assert all(torch.equal(before[k],v) for k,v in after.items()) and net.export()==raw
        loss=after['up1_low_pw'][:,:14].square().mean();loss.backward();g=net.weight['up1_low_pw'].grad
        assert torch.isfinite(g).all() and g.norm()>0
    print('READOUT_PREPARED nodes0/constant1215/20KiB/finite_nonzero_new_gradient',flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    if '--prepare' in sys.argv:
        ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True)
        prepare(ap.parse_args())
    else:driver.main(training_factory=TrainableLumaReadoutTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
