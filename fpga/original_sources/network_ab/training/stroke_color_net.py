"""Native contour features plus a learned color renderer, within nominal B1 MACs.

This is a NEW graph, not an FPGA deployment claim. The edge prediction enters
the renderer, and supervision also checks visible edges in the resulting RGB.
No source bypass, palette, external detector, or post-filter is used.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .native_contour_net import Block
from .train_context_contours import ContextContourNet


class StrokeColorNet(ContextContourNet):
    def __init__(self, difference=True):
        super().__init__(difference)
        self.color_context=nn.Conv2d(32,8,1)
        self.ink_features=nn.Conv2d(1,8,1)
        self.render=Block(8,8,difference=False)
        self.rgb=nn.Conv2d(8,3,1)
        with torch.no_grad():
            self.color_context.weight.zero_();self.color_context.bias.fill_(.3)
            self.ink_features.weight.zero_();self.ink_features.bias.zero_()
            self.render.dw.weight.zero_();self.render.dw.weight[:,0,1,1]=1
            self.render.dw.bias.zero_();self.render.pw.weight.zero_()
            for i in range(8):self.render.pw.weight[i,i,0,0]=1
            self.render.pw.bias.zero_();self.rgb.weight.mul_(.1);self.rgb.bias.zero_()

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
        features=F.relu(f+color+self.ink_features(cue))
        y=self.rgb(self.render(features))
        return (y,ink) if aux else y

    def budget(self):
        value=super().budget()
        value.update(complete_style_network=True,output='native720 signed RGB3',board_validated=False)
        return value
