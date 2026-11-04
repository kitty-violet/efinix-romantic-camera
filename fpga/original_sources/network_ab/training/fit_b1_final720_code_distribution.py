"""Same measured patch objective, real final720 clipped-code initialization.

The 0.05 outside-range gradient is training-only; every output remains the
ordinary rounded and clipped B1 result and is checked against its byte image.
"""
from . import fit_b1_unaligned_patch_distribution as runner
from .train_b1_trainable_luma_readout import ReadoutB1
from .train_b1_winner_surrogate import ExactValueProxyGradient


class Final720CodeB1(ReadoutB1):
    def training_activation_proxy(self,name,value,pre,row):
        if name=='up2_high_dw':
            clipped=pre.clamp(0,127)
            return ExactValueProxyGradient.apply(value,clipped+.05*(pre-clipped))
        return super().training_activation_proxy(name,value,pre,row)


if __name__=='__main__':
    runner.ReadoutB1=Final720CodeB1
    runner.main()
