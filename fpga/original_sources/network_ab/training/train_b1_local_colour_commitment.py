"""Learn decisive local RGB regions with the unchanged B1 forward graph."""
import copy
from pathlib import Path
import numpy as np,torch
from .train_b1_complementary_final_native import ComplementaryFinalTraining
from .train_b1_fine_rgb_native import evaluate as base_evaluate
from .train_b1_late_region_supervision import main as driver
from .train_b1_real_region_texture import align_pair
from .b1_local_colour_commitment import colour_commitment,regression
from .runtime import file_hash
from . import train_b1_minimum_area as m


class LocalColourCommitmentTraining(ComplementaryFinalTraining):
    def __init__(self,args,data):
        super().__init__(args,data)
        warm=torch.load(args.warm,map_location='cpu',weights_only=True)
        self.warm_provenance={k:warm[k] for k in ('manual_decoder_change','parent_checkpoint','parent_sha256','critic_initialization') if k in warm}
        self.hard_hashes={r['hard_region_target']:r['native_luma_colour_source_sha256'] for r in data['training']}
        assert all(file_hash(path)==sha for path,sha in self.hard_hashes.items())

    def batch_style(self,native):
        x,t,specs,flow,valid=super().batch_style(native);hard=[];original_labels=[]
        for spec in specs:
            row=self.by_id.get(spec['id']) or self.by_stem[spec['id']];xx,yy=spec['xy'];w,h=spec['wh']
            rgb=m.t.r.read(row['hard_region_target'])[yy:yy+h,xx:xx+w]
            lab=m.load_region(row['region'])[yy:yy+h,xx:xx+w]
            if spec['flip']:rgb=rgb[:,::-1];lab=lab[:,::-1]
            hard.append(torch.from_numpy(rgb.transpose(2,0,1).copy()).to(x.device).float()/128-1)
            original_labels.append(torch.from_numpy(lab.astype(np.int64).copy()).to(x.device))
        hard=torch.stack(hard);original_labels=torch.stack(original_labels)
        if flow is not None:
            # Identical geometry transport/reliability as the main soft target.
            hard,check,_,_=align_pair(x,hard,original_labels,original_labels,flow,valid)
            assert torch.equal(check,self.labels)
        else:assert torch.equal(original_labels,self.labels)
        self.hard_target=hard.detach()
        return x,t,specs,flow,valid

    def additional_loss(self,p,t,native):
        loss,values=super().additional_loss(p,t,native)
        hard=self.hard_target if native else self.hard_target[...,40:-40,40:-40]
        labels=self.labels if native else self.labels[...,40:-40,40:-40]
        if native:
            yy=self.rng.randrange(0,p.shape[-2]-384+1);xx=self.rng.randrange(0,p.shape[-1]-384+1)
            crop=lambda z:z[...,yy:yy+384,xx:xx+384]
            part,detail=colour_commitment(crop(p),crop(t),crop(hard),crop(labels))
        else:part,detail=colour_commitment(p,t,hard,labels)
        main=torch.autograd.grad(loss,p,retain_graph=True)[0].detach().norm()
        side=torch.autograd.grad(part,p,retain_graph=True)[0].detach().norm()
        ramp=min(1.,self.step/200)
        coefficient=(.75*ramp*main/side.clamp_min(1e-12)).clamp_max(16.)
        values.update(detail,colour_commitment_coefficient=coefficient,
            colour_commitment_actual_rgb_gradient_ratio=coefficient*side/main.clamp_min(1e-12))
        return loss+coefficient*part,values

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(self.warm_provenance)
        state.update(local_colour_commitment=dict(radius720=4,colour_selection='one joint RGB triplet from local hard target',
            original_AA_allowance=True,gradient_fraction=.75,coefficient_cap=16,ramp_steps=200,
            hard_target_sha256=copy.deepcopy(self.hard_hashes),training_only=True),visual_acceptance=False)
        torch.save(state,path)

    def preflight_extras(self):
        out=super().preflight_extras();self.step=200;row=self.update(False)
        assert row['temporal_pair'] and row['values']['colour_commitment_coefficient']>0
        assert 0<row['values']['colour_commitment_actual_rgb_gradient_ratio']<=.750001
        out['full_colour_commitment_and_real_flow']=row
        return out


@torch.inference_mode()
def evaluate(net,rows):
    result=base_evaluate(net,rows)
    for row,record in zip(rows,result['records']):
        x=m.t.r.tensor(m.t.r.read(row['path']))[None]
        t=m.t.r.tensor(m.t.r.read(row['target']))[None]
        hard=m.t.r.tensor(m.t.r.read(row['hard_region_target']))[None]
        labels=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None].to(x.device)
        value,_=colour_commitment(net(x)[0],t,hard,labels)
        record['local_rgb_colour_commitment']=float(value)
    result['mean']['local_rgb_colour_commitment']=sum(r['local_rgb_colour_commitment'] for r in result['records'])/len(rows)
    return result


if __name__=='__main__':
    torch.set_num_threads(2);regression()
    driver(training_factory=LocalColourCommitmentTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=m.t.preview)
