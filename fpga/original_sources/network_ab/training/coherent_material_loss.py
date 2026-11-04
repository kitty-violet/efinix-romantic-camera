"""Training-only local feature matching to reviewed game imagery.

Nearest-neighbor cosine matching is inspired by ARF (ECCV 2022):
https://github.com/Kai-46/ARF-svox2/blob/master/opt/nnfm_loss.py
This independent VGG19 implementation does not reproduce its 3D pipeline.
"""
import torch
from torch.nn import functional as F

LEVELS=('relu2_2','relu3_3')


def texture_view(image):
    """Remove image-wide gray offset/gain, retaining local spatial patterns."""
    y=(image*image.new_tensor([.2126,.7152,.0722])[None,:,None,None]).sum(1,keepdim=True)
    mean=y.mean((2,3),keepdim=True)
    std=y.std((2,3),keepdim=True,unbiased=False).clamp_min(.025)
    view=(.2*(y-mean)/std+.5).clamp(.01,.99)
    return (view*255/128-1).repeat(1,3,1,1)


def feature_rows(feature,stride=4,margin=3):
    if min(feature.shape[-2:])<=margin*2:raise ValueError('Feature field too small')
    return F.normalize(feature[:,:,margin:-margin:stride,margin:-margin:stride].permute(0,2,3,1).reshape(-1,feature.shape[1]),dim=1,eps=1e-6)


def nearest_cosine_loss(query,bank,chunk=256):
    """The selected target stays in autograd's value path; bank is frozen."""
    if query.ndim!=2 or bank.ndim!=2 or query.shape[1]!=bank.shape[1] or not len(bank):
        raise ValueError('Incompatible feature bank')
    values=[]
    for part in query.split(chunk):
        # Avoid retaining a large query x bank graph; indices need no gradient.
        with torch.no_grad():indices=(part.detach()@bank.T).argmax(1)
        values.append(1-(part*bank[indices]).sum(1))
    return torch.cat(values).mean()


def material_loss(features,bank):
    return torch.stack([nearest_cosine_loss(feature_rows(features[k]),bank[k]) for k in LEVELS]).mean()
