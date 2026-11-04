"""Continue actual B1 learning with the self-consistent RGB boundary loss."""
from pathlib import Path
import numpy as np
import torch
from .train_b1_ordered_distance_rims import OrderedDistanceRimTraining,driver,preview,rgb_evaluate
from .train_b1_reference_boundary_profile import reference_boundary_profile,regression
from . import train_b1_minimum_area as m


class ReferenceProfileDistanceTraining(OrderedDistanceRimTraining):
    boundary_profile_loss=staticmethod(reference_boundary_profile)

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(rgb_profile_target_residual_preserved=True,rgb_profile_equal_target_penalty=0,
            profile_span720=10,profile_shift720=2,visual_acceptance=False)
        torch.save(s,path)


@torch.inference_mode()
def evaluate(net,rows):
    result=rgb_evaluate(net,rows)
    for record,row in zip(result['records'],rows):
        x=m.t.r.tensor(m.t.r.read(row['path']))[None];target=m.t.r.tensor(m.t.r.read(row['target']))[None]
        labels=torch.as_tensor(m.load_region(row['region']).astype(np.int64),device=x.device)[None]
        _,v=reference_boundary_profile(net(x)[0],target,labels)
        # base evaluator adds the RGB-profile term only to the mean, while
        # per-record score still means joint-region loss. Reconstruct its
        # weighted selection score explicitly, rather than subtracting a term
        # that was never in the per-record score.
        record['previous_chord_only_profile_shape']=record['rgb_profile_shape']
        record.update({k:float(a) for k,a in v.items()})
        record['base_joint_score']=record['score']
        record['score']=(.7*record['base_joint_score']+4*record['region_flatness']+
            record['adjacent_region_contrast']+4*record['region_tail_flatness']+
            4*(record['rgb_profile_shape']+record['rgb_profile_contrast']))
    for key in ('rgb_profile_shape','rgb_profile_contrast','profile_samples','previous_chord_only_profile_shape','score'):
        result['mean'][key]=sum(r[key] for r in result['records'])/len(rows)
    return result


if __name__=='__main__':
    import sys
    torch.set_num_threads(2);regression()
    if '--check' not in sys.argv:
        driver.main(training_factory=ReferenceProfileDistanceTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
