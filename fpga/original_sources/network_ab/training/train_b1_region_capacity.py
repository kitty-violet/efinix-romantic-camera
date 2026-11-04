"""Use B1's existing residual capacity while keeping region formation fixed.

The previous explicit experiment tied24 channels to three coordinates and
froze all residual computation. Here only the final region/morphology/colour
head is fixed. No new inference operation or external mask is introduced.
"""
from pathlib import Path
import torch
from .train_b1_explicit_regions import ExplicitRegionTraining
from . import train_b1_late_region_supervision as driver
from . import train_b1_region_formation as formation


class RegionCapacityTraining(ExplicitRegionTraining):
    def __init__(self,args,data):
        super().__init__(args,data)
        assert self.field_node=='res2_add'
        for key in self.net.weight:
            if key.startswith(('res1_','res2_')):
                self.net.weight[key].requires_grad_(True)
                self.net.bias[key].requires_grad_(True)
                self.fixed.pop(key,None)
        # Live unused features feed initially zero outgoing coefficients.
        # Their revival must leave RGB inference exactly unchanged.
        with torch.no_grad():
            sample=torch.linspace(-1,.99,3*64*64,device='cuda').reshape(1,3,64,64)
            before=self.net(sample)[0].clone()
            if not self.net.weight['down2__pw'][24:].count_nonzero():
                for c in range(24,32):
                    src=(c%3)*4
                    self.net.weight['down2__pw'][c]=self.net.weight['down2__pw'][src]*.25
                    self.net.bias['down2__pw'][c]=32+self.net.bias['down2__pw'][src]*.25
                for stage in (1,2):
                    for half in (1,2):
                        key=f'res{stage}_conv{half}__dw'
                        self.net.weight[key][24:].zero_();self.net.weight[key][24:,0,1,1]=1
                        self.net.bias[key][24:]=0
                    key=f'res{stage}_conv1__pw'
                    self.net.weight[key][24:].zero_();self.net.bias[key][24:]=0
                    for c in range(24,32):self.net.weight[key][c,c,0,0]=1
                keys=['down2__pw']+[f'res{s}_conv{h}__dw' for s in (1,2) for h in (1,2)]+[f'res{s}_conv1__pw' for s in (1,2)]
                for key in keys:
                    w=self.net.weight[key][24:]
                    scale=torch.pow(2,torch.ceil(torch.log2(w.abs().flatten(1).amax(1).clamp_min(1e-5)/127)))
                    getattr(self.net,key+'_m')[:,24:]=(scale*131072).clamp(1,131071).double()[None,:,None,None]
            after=self.net(sample)[0]
            err=float((before-after).abs().max());assert err==0,err
            self.revival_rgb_error=err
        print('REGION_CAPACITY',sum(p.numel() for p in self.net.parameters() if p.requires_grad),
              'REVIVAL_RGB_ERROR',self.revival_rgb_error,flush=True)

    def project(self):
        # Do not collapse the learned32-channel state back to three copies.
        pass


if __name__=='__main__':
    driver.main(training_factory=RegionCapacityTraining,
                extra_sources=[Path(__file__),Path(formation.__file__),Path(__file__).with_name('train_b1_explicit_regions.py')],
                evaluation=formation.evaluate)
