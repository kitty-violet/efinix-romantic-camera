"""Actual-output colour decision confusion, CPU only."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from .train_b1_learned_colour import ColourB1
from .probe_b1_cartoon_int8 import IntegerB1
from .b1_trained_image import runtime_from_image
from .probe_b1_decoder_coverage import fragment
from .build_b1_ordered_palette import bits,GROUPS
from .fit_b1_sixteen_regions import region_class_map,palette_key
from .runtime import write_json


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);root=Path(__file__).resolve().parents[2]/'model-training-workspace/runs'
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    n=ColourB1(revive=False,source=root/'b1-identity-reconstruction-20261005/initial_identity.bin');n.load_state_dict(state['model'])
    n.package['residual_encoding']=state['residual_encoding'];n.quantized=True
    z=torch.zeros(200,16,1,1);z[:,:15,0,0]=bits('cpu')*127
    pal=fragment(n,z,'up2_low_dw')['rgb_head'][:,:,0,0]/128;pk=palette_key(pal)
    d=json.loads((root/'b1-minimum-area-20261006/data.json').read_text(encoding='utf-8'))
    actual=IntegerB1(runtime_from_image(n.export()),device='cpu');records=[]
    for row in d['validation']:
        if row['id'] not in ('val_stairs_090','val_plaza_210'):continue
        rgb=np.array(Image.open(row['path']).convert('RGB'));x=torch.from_numpy(rgb.transpose(2,0,1)[None].copy()).float()-128
        nodes,_=actual(x,True);out=nodes['up2_low_dw'][0,:15]>63.5
        ids=torch.from_numpy(region_class_map(row['target'],row['region'],pk).astype(np.int64))
        wanted=F.avg_pool2d(bits('cpu')[ids].permute(2,0,1)[None],2)[0]>=.5
        rec=[]
        for i in range(15):
            tp=float((out[i]&wanted[i]).sum());tn=float((~out[i]&~wanted[i]).sum())
            wrong=out[i]!=wanted[i];field=nodes['up1_high_dw'][0,i]
            endpoint=(field==0)|(field==127)
            rec.append(dict(bit=i,target_positive=float(wanted[i].float().mean()),actual_positive=float(out[i].float().mean()),
                positive_recall=tp/max(1,float(wanted[i].sum())),negative_recall=tn/max(1,float((~wanted[i]).sum())),
                errors_at_activation_endpoints=float((wrong&endpoint).sum())/max(1,float(wrong.sum())),
                field_min=float(field.min()),field_max=float(field.max())))
        groups=[]
        for start,end in GROUPS:
            truth_count=wanted[start:end].sum(0);out_count=out[start:end].sum(0)
            confusion=torch.bincount((truth_count*(end-start+1)+out_count).flatten(),minlength=(end-start+1)**2).reshape(end-start+1,end-start+1)
            groups.append(dict(counts=confusion.tolist(),accuracy=float((truth_count==out_count).float().mean())))
        records.append(dict(id=row['id'],bits=rec,groups=groups));print('BITS',row['id'],rec,flush=True)
    write_json(a.out,dict(checkpoint=a.checkpoint,records=records,actual_bytes=True))


if __name__=='__main__':main()
