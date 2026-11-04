"""Same final RGB objective, with complete thin/multi-region profiles."""
from .train_b1_full_rgb_geometry import luma_regions
from .train_b1_luma_chroma_interfaces import chroma_objective
from .b1_complete_rgb_profiles import complete_rgb_profiles
from .audit_b1_rim_tolerance import rim_terms


class CompleteRGBGeometryTraining:
    def additional_loss(self,p,t,native):
        labels=self.labels if native else self.labels[...,40:-40,40:-40]
        colour,v=chroma_objective(p,t,labels);luma,l=luma_regions(p,t,labels)
        profile,detail=complete_rgb_profiles(p,t,labels);rim,_=rim_terms(p,t,labels,2)
        v.update(l);v.update(detail);v['two_pixel_region_centred_bright_shoulder']=rim
        return colour+luma+4*profile+rim,v
