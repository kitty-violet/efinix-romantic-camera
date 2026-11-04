"""Add a learned color entrance; all colors still pass through the renderer."""
import torch
from torch import nn
from torch.nn import functional as F
from .stroke_color_net import StrokeColorNet


class StrokeColorInputNet(StrokeColorNet):
    def __init__(self,difference=True):
        super().__init__(difference)
        self.color_input=nn.Conv2d(3,8,1)
        with torch.no_grad():
            self.color_input.weight.zero_();self.color_input.bias.fill_(1.)
            for i in range(3):
                self.color_input.weight[i,i,0,0]=2
                self.color_input.weight[i+3,i,0,0]=-2

    def forward(self,x,aux=False,ablate_ink=False):
        native=self.native(F.relu(self.stem(x)))
        half=self.half_block(self.down1(native));quarter=self.down2(half)
        c=self.down3(quarter)
        for block in self.context:c=c+block(c)
        c=F.interpolate(c,scale_factor=2,mode='nearest')+quarter
        h=self.decode1(F.interpolate(F.relu(self.reduce(c)),scale_factor=2,mode='nearest')+half)
        f=F.relu(self.finish(F.interpolate(h,scale_factor=2,mode='nearest')+native))
        ink=self.edge(f)
        color=F.interpolate(self.color_context(c),scale_factor=4,mode='nearest')
        cue=torch.zeros_like(ink) if ablate_ink else ink/8
        features=F.relu(f+color+self.ink_features(cue)+self.color_input(x))
        y=self.rgb(self.render(features))
        return (y,ink) if aux else y
