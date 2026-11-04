"""Test whether the current texture objective rewards tone-only changes.

No model update or dataset mutation. Fixed diagnostic transforms are not results.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .coherent_guarded_train import TextureTraining,gray,centered_gram,TEXTURE_LEVELS
from . import coherent_shared_train as base
from .runtime import file_hash,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    config=json.loads(a.config.read_text(encoding='utf-8'));trainer=TextureTraining(config)
    state=torch.load(a.weights,map_location='cpu',weights_only=True)
    trainer.net.load_state_dict(state['generator']);trainer.net.eval()
    records=[]
    with torch.no_grad():
        for index in range(12):
            x,_,_,_,_,spec=trainer.samples.next();y=trainer.net(x)
            lum=gray(y);smooth=F.avg_pool2d(F.pad(lum,(2,)*4,mode='reflect'),5,stride=1)
            variants={'unchanged':y,'luma_minus12':y-12/128,'luma_plus12':y+12/128,
                      'contrast_080':y.mean((2,3),keepdim=True)+.8*(y-y.mean((2,3),keepdim=True)),
                      'fine_luma_half':y-.5*(lum-smooth)}
            measurements={}
            for name,value in variants.items():
                value=value.clamp(-1,127/128)
                feats=trainer.vgg.extract(gray(value).repeat(1,3,1,1))
                style=torch.stack([base.relative_mse(centered_gram(feats[k]),trainer.texture_bank[k])
                                   for k in TEXTURE_LEVELS]).mean()
                measurements[name]={'centered_gray_vgg_loss':float(style)}
            # The newly constructed discriminator is untrained: do not evaluate or use its scores.
            records.append({'index':index,'training_sample':spec,'measurements':measurements})
    names=list(records[0]['measurements']);summary={}
    for name in names:
        values=[r['measurements'][name]['centered_gray_vgg_loss'] for r in records]
        baseline=[r['measurements']['unchanged']['centered_gray_vgg_loss'] for r in records]
        summary[name]={'mean_loss':float(np.mean(values)),
             'samples_improved_vs_unchanged':sum(v<b for v,b in zip(values,baseline))}
    result={'status':'READ_ONLY_LOSS_PROBE','samples':len(records),
      'weights_sha256':file_hash(a.weights),'config_sha256':file_hash(a.config),
      'summary':summary,'records':records,
      'interpretation':'Tone and blur probes are diagnostics, not trained improvements or acceptable style results.'}
    write_json(a.output,result);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
