"""Allow mixed boundary occupancies without tying every threshold row.

Initialize from shared-coordinate weights, but let the existing30 PW rows learn
different boundary responses. Supervise the ordered target occupancies and
penalize rank violations. Ordering is measured, not claimed as a hard guarantee.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from .train_b1_boundary_occupancy import BoundaryOccupancyTraining,driver,preview
from .train_b1_exact_complement_hinges import evaluate as base_evaluate
from .train_b1_predecision_regions import PredecisionTraining
from .train_b1_ternary_packed720 import Packed720Training
from .train_b1_ordinal_vote_hinges import OrdinalVoteB1
from .train_b1_quantum_regions import RowScaledAdam
from .build_b1_thirty_binary_regions import GROUPS30
from . import train_b1_minimum_area as m

def rank_loss(z):
    errors=torch.cat([F.relu(torch.diff(z[:,a:b],dim=1)) for a,b in GROUPS30],1)
    return errors.amax(1).mean()/127

class OrderedBoundaryTraining(BoundaryOccupancyTraining):
    def __init__(self,args,data):
        # Bypass CommonHingeTraining's gradient tying and post-update projection.
        # Preserve the exact decoder, data, optimizer units and active tensor API.
        PredecisionTraining.__init__(self,args,data)
        assert self.setup.get('ordinal_spatial_vote') and self.setup.get('exact_integer_complement')
        self.net.__class__=OrdinalVoteB1;self.fixed={};groups=[]
        for key,w in self.net.weight.items():
            fixed=key.startswith(('res3_conv2','up1_','up2_','rgb_head')) or key in ('down1__dw','down2__dw')
            b=self.net.bias[key];w.requires_grad_(not fixed);b.requires_grad_(not fixed)
            if fixed:self.fixed[key]=(w.detach().clone(),b.detach().clone())
            else:
                s=(getattr(self.net,key+'_m')/131072).float().flatten()
                groups.extend([dict(params=[w],lr=args.lr,row_gain=(s*256)[:,None,None,None]),dict(params=[b],lr=.008)])
        self.opt=RowScaledAdam(groups);self.base_lrs=[g['lr'] for g in groups]

    def update(self,native=None):return Packed720Training.update(self,native)

    def additional_loss(self,p,t,native):
        loss,v=super().additional_loss(p,t,native);rank=rank_loss(self.nodes['res3_conv1.pw'])
        v['ordinal_rank_violation']=rank
        return loss+10*rank,v

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(common_coordinate_hinges=False,tied_hinge_rows=False,independent_ordinal_head=True,
          ordinal_rank_guarantee=False,ordinal_rank_loss_weight=10,visual_acceptance=False)
        torch.save(s,path)

@torch.inference_mode()
def evaluate(net,rows):
    result=base_evaluate(net,rows)
    for rec,row in zip(result['records'],rows):
        nodes=net(m.t.r.tensor(m.t.r.read(row['path']))[None],all_nodes=True)
        for key,label in (('res3_conv1.pw','before_vote'),('up1_low_dw','after_vote')):
            dif=torch.cat([torch.diff(nodes[key][:,a:b],dim=1) for a,b in GROUPS30],1)
            rec['rank_'+label+'_mean']=float(dif.clamp_min(0).amax(1).mean())
            rec['rank_'+label+'_pixel_fraction']=float((dif>1).any(1).float().mean())
            rec['rank_'+label+'_max']=float(dif.clamp_min(0).max())
    for k in [k for k in result['records'][0] if k.startswith('rank_')]:result['mean'][k]=sum(r[k] for r in result['records'])/len(rows)
    result['mean']['score']+=10*result['mean']['rank_after_vote_mean']/127
    return result

def regression():
    z=torch.zeros(1,32,8,8);z[:,:14]=torch.arange(14,0,-1)[None,:,None,None]*8
    assert rank_loss(z)==0
    bad=z.clone();bad[:,4]=127;bad.requires_grad_();loss=rank_loss(bad);loss.backward()
    assert loss>0 and bad.grad[:,4].sum()>0 and bad.grad[:,3].sum()<0
    assert torch.isfinite(bad.grad).all()
    print('ORDERED_BOUNDARY_REGRESSION_PASS ordered0/violation_positive/directional_gradient',flush=True)

if __name__=='__main__':
    import sys
    torch.set_num_threads(2);regression()
    if '--check' not in sys.argv:driver.main(training_factory=OrderedBoundaryTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
