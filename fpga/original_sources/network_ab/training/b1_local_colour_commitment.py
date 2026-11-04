"""Training-only attraction to complete, nearby target RGB region colours.

The closest RGB triplet is chosen jointly. Independent R/G/B choices would
invent false edge colours. This nonconvex energy supplements geometry; it
does not prove topology, minimum area, temporal stability or sharpness alone.
"""
import torch
from torch.nn import functional as F


def colour_commitment(p,target,hard,labels):
    n,c,h,w=p.shape
    assert p.shape==target.shape==hard.shape and labels.shape==(n,h,w)
    # The allowed region colours can come from a displacement of <=4 pixels.
    # A common RGB candidate index is used for all channels.
    pad=F.pad(hard,(4,4,4,4),mode='replicate')
    candidate=[pad[...,4+dy:4+dy+h,4+dx:4+dx+w] for dy in (-4,-2,0,2,4) for dx in (-4,-2,0,2,4)]
    seam=torch.zeros_like(labels,dtype=torch.bool)
    seam[:,:,1:]|=labels[:,:,1:]!=labels[:,:,:-1];seam[:,1:]|=labels[:,1:]!=labels[:,:-1]
    band=F.max_pool2d(seam[:,None].float(),9,1,4)>0
    band[...,:8,:]=False;band[...,-8:,:]=False;band[...,:,:8]=False;band[...,:,-8:]=False
    with torch.no_grad():
        # Selection and penalty use the same distance. Selecting with L2 but
        # allowing the minimum Huber cost would falsely penalize p==target.
        costs=torch.stack([F.smooth_l1_loss(p.detach(),v,beta=2/128,reduction='none').mean(1) for v in candidate],1)
        index=costs.argmin(1)
        selected=torch.zeros_like(p)
        for j,v in enumerate(candidate):selected+=v*(index==j)[:,None]
        # Correct antialiased target pixels retain their existing allowance.
        # This is not a demand for nearest-neighbour jagged output.
        tolerance=torch.stack([F.smooth_l1_loss(target,v,beta=2/128,reduction='none').mean(1,keepdim=True) for v in candidate]).amin(0)
    residual=F.smooth_l1_loss(p,selected,beta=2/128,reduction='none').mean(1,keepdim=True)
    excess=F.relu(residual-tolerance)
    value=(excess*band).sum()/band.sum().clamp_min(1)
    return value,dict(local_rgb_colour_commitment=value,local_colour_boundary_fraction=band.float().mean())


def regression():
    torch.manual_seed(47);h,w=64,96
    labels=torch.zeros(1,h,w,dtype=torch.long);labels[...,48:]=1
    hard=torch.zeros(1,3,h,w);hard[:,0,:,:48]=.8;hard[:,1,:,48:]=.8
    target=F.avg_pool2d(F.pad(hard,(1,1,1,1),mode='replicate'),3,1)
    assert colour_commitment(target,target,hard,labels)[0]==0
    blur=F.avg_pool2d(F.pad(hard,(5,5,5,5),mode='replicate'),11,1).detach().requires_grad_()
    value,_=colour_commitment(blur,target,hard,labels);value.backward()
    assert value>.01 and torch.isfinite(blur.grad).all()
    # Each side is pushed toward one actual complete colour; invented yellow
    # cannot receive zero cost by independently matching red and green.
    assert blur.grad[0,0,32,45]<0 and blur.grad[0,1,32,45]>0
    yellow=hard.clone();yellow[...,45:51]=torch.tensor([.8,.8,0.])[None,:,None,None]
    assert colour_commitment(yellow,target,hard,labels)[0]>.01
    correct=hard.clone().requires_grad_();zero,_=colour_commitment(correct,target,hard,labels);assert zero==0
    zero.backward();assert torch.isfinite(correct.grad).all()
    assert colour_commitment(target,target,hard,labels*0)[0]==0
    palette=torch.rand(7,3)*1.8-.9;lab=torch.randint(7,(1,h,w))
    hard_random=palette[lab].permute(0,3,1,2)
    target_random=F.avg_pool2d(F.pad(hard_random,(1,1,1,1),mode='replicate'),3,1)
    assert colour_commitment(target_random,target_random,hard_random,lab)[0]==0
    print('LOCAL_RGB_COMMITMENT_PASS target_AA0/hard0/blur_direction/joint_RGB_no_yellow/empty0/finite',flush=True)


if __name__=='__main__':torch.set_num_threads(2);regression()
