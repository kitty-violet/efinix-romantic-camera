"""One explicit final-RGB objective with regional structure as the priority.

Weights follow the saved fixed-TRAIN component-gradient audit, not validation
image tuning. Flatness/shoulder/turn penalties remain guards, not the main
geometry objective. Neither this loss nor labels run on the FPGA.
"""
from .b1_all_region_means import all_region_colour_objective
from .b1_complete_rgb_profiles import complete_rgb_profiles
from .b1_boundary_palette_distribution import palette_distribution
from .audit_b1_rim_tolerance import rim_terms


WEIGHTS=dict(final_luma_pixel=1,all_region_luma_mean=4,final_luma_region_error_variation=1,
    chroma_pixel=2,all_region_chroma_mean=4,chroma_region_residual_variation=1,
    complete_rgb_profile=4,boundary_palette_distribution=8,
    two_pixel_region_centred_bright_shoulder=.25)


def geometry_first(p,t,labels):
    _,terms=all_region_colour_objective(p,t,labels)
    profile,pv=complete_rgb_profiles(p,t,labels)
    palette,sv=palette_distribution(p,t,labels)
    rim,_=rim_terms(p,t,labels,2)
    terms.update(pv);terms.update(sv);terms['two_pixel_region_centred_bright_shoulder']=rim
    return sum(terms[k]*w for k,w in WEIGHTS.items()),terms
