"""Inspect activations and code changes after painting-code transfer.

Read-only diagnostic on TRAIN sources; no target injection in delivered output.
"""
import argparse,copy,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .train_b1_learned_colour import ColourB1
from .runtime import write_json,file_hash


def load(path):
    state=torch.load(path,map_location='cpu',weights_only=True)
    net=ColourB1(revive=False);net.load_state_dict(state['model']);net.quantized=True
    net.package['residual_encoding']=copy.deepcopy(state['residual_encoding'])
    return net.eval()


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--student',required=True);ap.add_argument('--teacher',required=True)
    ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
    torch.set_num_threads(2)
    s,t=load(args.student),load(args.teacher)
    d=json.loads(Path(args.data).read_text(encoding='utf-8'));rows=[];seen=set()
    for row in d['training']:
        if row['video'] not in seen:rows.append(row);seen.add(row['video'])
    records=[]
    for row in rows:
        def tensor(path):
            a=np.asarray(Image.open(path).convert('RGB')).copy();h,w=a.shape[:2]
            y=(h//2-128)//4*4;x=(w//2-128)//4*4
            return torch.from_numpy(a[y:y+256,x:x+256].transpose(2,0,1).copy())[None].float()/128-1
        x,y=tensor(row['path']),tensor(row['target']);sn=s(x,all_nodes=True);tn=t(y,all_nodes=True)
        nodes={}
        for name,a in sn.items():
            b=tn[name];n=a.flatten(2)
            interior=((n>0)&(n<127)).float().mean(-1).flatten()
            nodes[name]=dict(mae=float((a-b).abs().mean()),std=float(n.std(-1).mean()),
                zero=float((a==0).float().mean()),negative=float((a<0).float().mean()),sat127=float((a==127).float().mean()),
                always_interior_rows=int((interior>.995).sum()),channels=a.shape[1],
                target_std=float(b.flatten(2).std(-1).mean()))
        records.append(dict(id=row['id'],video=row['video'],nodes=nodes))
    changes={}
    for k,w in s.weight.items():
        scale=(getattr(s,k+'_m')/131072).float().flatten()
        ws=(w/scale[:,None,None,None]).round().clamp(-127,127)
        wt=(t.weight[k]/scale[:,None,None,None]).round().clamp(-127,127)
        changes[k]=dict(changed_fraction=float((ws!=wt).float().mean()),mean_code_delta=float((ws-wt).abs().mean()),
            scale_min=float(scale.min()),scale_max=float(scale.max()))
    write_json(Path(args.out),dict(student_sha256=file_hash(args.student),teacher_sha256=file_hash(args.teacher),
        scope='Seven TRAIN source-centre crops, actual source prediction vs target encoding; diagnostic only',records=records,changes=changes))
    keys=['stem','down1.pw','down2.pw','res1_add','res2_add','res3_add','up2_low_pw','rgb_head']
    print(json.dumps({k:{field:sum(r['nodes'][k][field] for r in records)/len(records) for field in records[0]['nodes'][k]} for k in keys},indent=2),flush=True)


if __name__=='__main__':main()
