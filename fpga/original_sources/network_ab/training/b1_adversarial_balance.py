"""Balance a training critic against the objective that actually runs.

Uses RGB-output gradients, before B1 parameter Jacobians/optimizer updates.
It is not a guarantee of equal parameter updates or visual correctness.
"""
import torch


def actual_output_balance(reference,prediction,adversarial,critic_input,amount,progress):
    main=torch.autograd.grad(reference,prediction,retain_graph=True)[0].detach()
    style=torch.autograd.grad(adversarial,critic_input,retain_graph=True)[0].detach()
    main_norm=main.norm();style_norm=style.norm()
    factor=(amount*min(max(progress,0.),1.)*main_norm/style_norm.clamp_min(1e-12)).clamp_max(4.)
    actual_ratio=factor*style_norm/main_norm.clamp_min(1e-12)
    return factor,dict(critic_reference_output_gradient=main_norm,
                      critic_unscaled_output_gradient=style_norm,
                      critic_actual_output_gradient_ratio=actual_ratio)


def regression():
    torch.manual_seed(37)
    p=torch.randn(2,3,16,20,requires_grad=True);target=torch.randn_like(p)
    reference=7*(p-target).square().mean();part=p[...,4:12,6:14]
    adv=part.sin().mean()
    for fraction in (0.,.25,.5):
        factor,values=actual_output_balance(reference,p,adv,part,.5,fraction)
        assert torch.isfinite(factor) and not factor.requires_grad
        assert abs(float(values['critic_actual_output_gradient_ratio'])-.5*fraction)<1e-6
        ga=torch.autograd.grad(factor*adv,p,retain_graph=True)[0]
        gm=torch.autograd.grad(reference,p,retain_graph=True)[0]
        assert abs(float(ga.norm()/gm.norm())-.5*fraction)<1e-6
    factor,_=actual_output_balance(p.sum()*0,p,adv,part,.5,1.)
    assert factor==0
    print('ACTUAL_OUTPUT_BALANCE_PASS full_output/crop_gradient_ratio/warmup/detached/zero_reference',flush=True)


if __name__=='__main__':regression()
