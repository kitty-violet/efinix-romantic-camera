"""Reduce the measured4px shoulder allowance while keeping region brightness.

The2px radius is justified by the3.30px ideal10--90 contour width. Coefficient1
keeps this term below the primary centered gradient in the seven-source audit;
coefficient4 would dominate some samples and is not used. All operations here
are training-only; the exact14Y+continuous chroma B1 decoder is unchanged.
"""
from pathlib import Path
import numpy as np,torch
from .train_b1_direct_luma_interfaces import DirectLumaInterfaceTraining,evaluate as base_evaluate,driver,preview
from .audit_b1_rim_tolerance import rim_terms,regression
from . import train_b1_minimum_area as m


class TightLumaRimTraining(DirectLumaInterfaceTraining):
    def additional_loss(self,p,t,native):
        value,v=super().additional_loss(p,t,native)
        labels=self.labels if native else self.labels[...,40:-40,40:-40]
        tight,_=rim_terms(p,t,labels,2)
        v['two_pixel_region_centred_bright_shoulder']=tight
        return value+tight,v

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(bright_rim_objective='region_mean_centred_radius2_top5percent',bright_rim_weight=1.,
            bright_rim_region_dc_gradient_zero=True,old_four_pixel_allowance_replaced=True,
            ideal_contour_10_90_width_measured=3.300229,extra_inference_operations=0,visual_acceptance=False)
        torch.save(s,path)


@torch.inference_mode()
def evaluate(net,rows):
    result=base_evaluate(net,rows)
    for rec,row in zip(result['records'],rows):
        x=m.t.r.tensor(m.t.r.read(row['path']))[None];t=m.t.r.tensor(m.t.r.read(row['target']))[None]
        labels=torch.from_numpy(m.load_region(row['region']).astype(np.int64))[None].to(x.device)
        value,_=rim_terms(net(x)[0],t,labels,2);rec['two_pixel_region_centred_bright_shoulder']=float(value)
    key='two_pixel_region_centred_bright_shoulder';result['mean'][key]=sum(r[key] for r in result['records'])/len(rows)
    # Keep the old selection score unchanged; a lower new diagnostic alone
    # must not turn an overall darker/blurred image into a passing candidate.
    return result


if __name__=='__main__':
    torch.set_num_threads(2);regression()
    driver.main(training_factory=TightLumaRimTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
