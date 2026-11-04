"""Build independent rational scales from every residual record in a B1 image.

Convolution output scales are chosen to reproduce the serialized alignment;
no residual branch flag or multiplier is assumed from an older checkpoint.
"""
from fractions import Fraction
from .runtime import contract_guard


def oracle_records(params,residual):
    scales={'input':Fraction(1,128)};out={};res={}
    ratio=lambda x:[x.numerator,x.denominator]
    for r in contract_guard()[1]['nodes']:
        name=r['id'];si=scales[r['inputs'][0]]
        if r['kind']=='nearest_x2':scales[name]=si;continue
        if r['kind']=='residual_add':
            sk=scales[r['inputs'][1]];common=max(si,sk);q=residual[name]
            so=common*Fraction(131072,q['post_multiplier'])
            assert (si<sk)==q['main_scaled']
            assert min(si,sk)/common==Fraction(q['multiplier'],131072)
            res[name]=dict(main_scale=ratio(si),skip_scale=ratio(sk),common_scale=ratio(common),output_scale=ratio(so))
        else:
            so=si
            if name.startswith('res') and name.endswith('conv2.pw'):
                stage=int(name[3]);skip='down2.pw' if stage==1 else f'res{stage-1}_add'
                q=residual[f'res{stage}_add'];factor=Fraction(q['multiplier'],131072)
                so=scales[skip]*(factor if q['main_scaled'] else 1/factor)
            if name=='rgb_head':so=Fraction(1,128)
            p=params[name]
            out[name]=dict(weights=[list(a) for a in p.weights],bias=list(p.bias),multipliers=list(p.multiplier),
                input_scale=ratio(si),output_scale=ratio(so),weight_scales=[ratio(Fraction(m,131072)*so/si) for m in p.multiplier])
        scales[name]=so
    return out,res
