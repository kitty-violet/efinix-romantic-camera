"""Fix game correspondences from the stable anchor, so blur cannot rematch."""
import torch
from .coherent_material_loss import LEVELS,feature_rows


@torch.no_grad()
def reference_targets(features,bank):
    targets={}
    for k in LEVELS:
        query=feature_rows(features[k])
        ids=torch.cat([(part@bank[k].T).argmax(1) for part in query.split(256)])
        targets[k]=bank[k][ids]
    return targets


def fixed_material_loss(features,targets):
    values=[]
    for k in LEVELS:
        query=feature_rows(features[k])
        if query.shape!=targets[k].shape:raise ValueError('Correspondence shape changed')
        values.append((1-(query*targets[k]).sum(1)).mean())
    return torch.stack(values).mean()
