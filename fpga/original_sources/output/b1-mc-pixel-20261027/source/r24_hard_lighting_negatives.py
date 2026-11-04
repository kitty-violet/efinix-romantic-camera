"""Continue MC support learning on high-confidence false lighting examples.

Mine only freshly generated training samples, never review videos. Keep an
independent validation generator and use a fresh seed for the next final audit.
"""
from core import *
import r18_learn_structure_support as fit
import argparse

original_samples=fit.samples
active=None

def start():
    global active
    active=load_model(OUT/'r20-multifeature-structure/selected.pt');active.support_rows=slice(23,32)
    active.support_down_rows=slice(3,8);active.training_saturation_leak=.02
    return active

@torch.no_grad()
def mined_samples(count,seed,size=128):
    if seed!=271111:return original_samples(count,seed,size)
    arrays,labels,kinds=original_samples(1536,271140,size);scores=[]
    for base in range(0,len(arrays),8):
        x=torch.stack([tensor(a,'cpu') for a in arrays[base:base+8]]).cuda()
        pred=active(x,True)['up1_low_pw'][:,3,10:-10,10:-10]
        scores.extend(pred.float().mean((1,2)).cpu().tolist())
    scores=np.asarray(scores);rng=np.random.default_rng(271140)
    pos=np.where(kinds<2)[0];neg=np.where(kinds>=2)[0]
    hardest=neg[np.argsort(scores[neg])[-128:]];remaining=np.setdiff1d(neg,hardest)
    selected=np.r_[rng.choice(pos,128,replace=False),hardest,rng.choice(remaining,128,replace=False)]
    rng.shuffle(selected);assert len(selected)==count and len(np.unique(selected))==count
    d=OUT/'r24-hard-lighting-negatives';d.mkdir(exist_ok=True)
    save_json(d/'mining.json',dict(generator_seed=271140,pool=1536,selected_indices=selected.tolist(),
        hard_negative_count=128,random_negative_count=128,positive_count=128,
        hard_negative_score_mean=float(scores[hardest].mean()),random_negative_score_mean=float(scores[selected[kinds[selected]>=2]].mean()),
        review_videos_used_for_gradients=False,previous_final_test_used_for_training=False))
    print('MC_HARD_NEGATIVES_MINED',len(hardest),round(float(scores[hardest].mean()),3),flush=True)
    return [arrays[i] for i in selected],labels[selected],kinds[selected]

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=500);p.add_argument('--dest',default='r24-hard-lighting-negatives-v2');args=p.parse_args()
    fit.start=start;fit.samples=mined_samples
    fit.main(argparse.Namespace(steps=args.steps,warmup=0,dest=args.dest,lr_scale=.15))
