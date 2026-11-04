"""Balance inverse-code supervision against the measured hard-decoder gain.

On the fixed three TRAIN crops, current code-gradient norm at up1_low_pw is
1.202e-5 versus7.785e-4 for the rendered profile. A fixed64x coefficient
puts them at similar readout scale. This does not increase learning rate,
change the integer forward, or claim that gradient balance ensures success.
"""
from .train_b1_relaxed_count_interfaces import RelaxedCountInterfaceTraining


class BalancedInverseGeometryTraining(RelaxedCountInterfaceTraining):
    def additional_loss(self,p,t,native):
        loss,v=super().additional_loss(p,t,native)
        code=.5*v['ordered_distance_full']+.25*v['ordered_distance_band']
        v['inverse_code_extra_weighted_loss']=63*code
        return loss+63*code,v
