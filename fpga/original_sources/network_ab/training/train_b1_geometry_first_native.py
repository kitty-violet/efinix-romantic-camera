"""Correct narrow-region coverage and stop auxiliary smoothing dominating.

The fixed B1 graph, real-source input, phase supervision, shift/noise guards,
source-flow temporal supervision and synthetic shape schedule are retained.
"""
from pathlib import Path
import numpy as np,torch
from .train_b1_complementary_final_native import ComplementaryFinalTraining
from .train_b1_fine_rgb_native import evaluate as historical_evaluate
from .b1_geometry_first_objective import geometry_first,WEIGHTS
from .b1_all_region_means import regression as mean_regression
from .b1_boundary_palette_distribution import regression as distribution_regression
from .train_b1_contour_coherence import contour_coherence
from .train_b1_late_region_supervision import main as driver
from . import train_b1_minimum_area as m


class GeometryFirstTraining(ComplementaryFinalTraining):
    contour_weight=.5
    rgb_objective=staticmethod(geometry_first)

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(geometry_first_objective=dict(weights=WEIGHTS,contour_weight=self.contour_weight,
            mean_coverage='all pixels in each region, including narrow/no-core regions',
            flatness_coverage='unchanged core-only',palette_distribution='13 fixed RGB projections,10px,shared +/-2px',
            selection_score='new objective; historical_score kept separately',extra_inference_operations=0),visual_acceptance=False)
        s['contour_coherence_training']=dict(window720=9,target_corner_allowance=.015,weight=self.contour_weight)
        s['contour_coherence_weight']=self.contour_weight
        torch.save(s,path)

    def preflight_extras(self):
        out=super().preflight_extras();v=out['update']['values']
        assert v['all_region_luma_mean']>0 and v['boundary_palette_distribution']>0
        assert v['palette_distribution_profiles']>0
        out['all_region_and_distribution_training_active']=True
        return out


@torch.inference_mode()
def evaluate(net,rows):
    result=historical_evaluate(net,rows)
    for row,rec in zip(rows,result['records']):
        x=m.t.r.tensor(m.t.r.read(row['path']))[None];t=m.t.r.tensor(m.t.r.read(row['target']))[None]
        lab=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None].to(x.device)
        p=net(x)[0];loss,terms=geometry_first(p,t,lab)
        curve=contour_coherence(p,t)
        rec['historical_score']=rec.pop('score');rec['score']=float(loss+.5*curve)
        rec.update({k:float(v) for k,v in terms.items()});rec['short_contour_turn_excess']=float(curve)
    result['mean']={k:sum(r[k] for r in result['records'])/len(rows) for k in result['records'][0] if k!='id'}
    return result


if __name__=='__main__':
    torch.set_num_threads(2);mean_regression();distribution_regression()
    driver(training_factory=GeometryFirstTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=m.t.preview)
