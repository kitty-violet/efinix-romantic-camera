"""Native720 contour curriculum with TRAIN-calibrated centered QAT updates."""
from pathlib import Path
import copy
import torch
from .train_b1_distance_curriculum import DistanceCurriculumTraining,evaluate,driver,preview
from .centered_b1_optimizer import CenteredB1Adam,calibration_means,regression
from . import train_b1_minimum_area as m


class CenteredDistanceTraining(DistanceCurriculumTraining):
    def __init__(self,args,data):
        super().__init__(args,data)
        self.centering_calibration_ids=[];sums={}
        with torch.no_grad():
            raw=self.net.export()
            for video in sorted({r['video'] for r in data['training']}):
                rows=sorted([r for r in data['training'] if r['video']==video],key=lambda r:r['id']);row=rows[len(rows)//2]
                self.centering_calibration_ids.append(row['id'])
                means=calibration_means(self.net,m.t.r.tensor(m.t.r.read(row['path']))[None])
                for k,v in means.items():sums[k]=sums.get(k,torch.zeros_like(v))+v
            means={k:v/len(self.centering_calibration_ids) for k,v in sums.items()}
            assert self.net.export()==raw
        self.opt=CenteredB1Adam([dict(g) for g in self.opt.param_groups],self.net,means)
        self.base_lrs=[g['lr'] for g in self.opt.param_groups]

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(centered_parameter_updates=True,centering_calibration_ids=self.centering_calibration_ids,
            centering_mean_fixed=True,quantized_weight_dc_bias_compensation=True,
            centering_before_gradient_clip=True,centering_extra_inference_operations=0,
            mean_calibration_validation_used=False,visual_acceptance=False)
        torch.save(s,path)


if __name__=='__main__':
    torch.set_num_threads(2);regression()
    driver.main(training_factory=CenteredDistanceTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
