"""Train the nonlinear region encoder through the tested painting decoder.

Controlled replacement of the failed reconstruction-derived encoder, not a
new graph. Preserve complete phase-bearing codec features, not30 averaged
palette bits. All quality previews use untouched source inputs.
"""
from pathlib import Path
import torch
from .train_b1_painting_codec import PaintingCodeTransferTraining
from .train_b1_quantum_regions import RowScaledAdam
from .train_b1_region_tail_flatness import region_tail_flatness,evaluate,preview
from .train_b1_joint_rgb_edges import rgb_edge_vector_loss
from . import train_b1_late_region_supervision as driver


class ShapeEncoderCodecTraining(PaintingCodeTransferTraining):
    def __init__(self,args,data):
        warm=torch.load(args.warm,map_location='cpu',weights_only=True)
        assert warm.get('shape_encoder_codec')
        super().__init__(args,data)
        groups=[]
        for key,w in self.net.weight.items():
            if not w.requires_grad:continue
            s=(getattr(self.net,key+'_m')/131072).float().flatten()
            groups.extend([dict(params=[w],lr=args.lr,**({'row_gain':(s*256)[:,None,None,None]} if args.qat else {})),
                           dict(params=[self.net.bias[key]],lr=.002)])
        self.opt=RowScaledAdam(groups);self.base_lrs=[g['lr'] for g in groups]

    def additional_loss(self,p,t,native):
        loss,values=super().additional_loss(p,t,native)
        labels=self.labels if native else self.labels[...,40:-40,40:-40]
        tail=region_tail_flatness(p,labels);vector=rgb_edge_vector_loss(p,t)
        values.update(region_tail_flatness=tail,joint_rgb_edge_vector=vector)
        return loss+2*tail+2*vector,values

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(shape_encoder_codec=True,visual_acceptance=False,
            prior_source_encoder='trained ternary region encoder',decoder='frozen painting codec',
            averaged_palette_bits=False,extra_inference_operations=0)
        torch.save(state,path)


if __name__=='__main__':
    driver.main(training_factory=ShapeEncoderCodecTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
