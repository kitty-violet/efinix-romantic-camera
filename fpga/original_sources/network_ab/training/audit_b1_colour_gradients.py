"""Measure saturation and parameter-space objective conflict, not just scores."""
import torch
from torch.nn import functional as F
from . import train_b1_learned_colour as colour
from . import train_b1_live_rank_features as live
from .runtime import configure_workspace,write_json
from .engine import set_reproducible


def main():
    configure_workspace();set_reproducible(2026100522)
    data,selected,cache=colour.load_data();tr=live.make(cache,selected,3000)
    tr.restore(live.OUT/'checkpoint_003000.pt');records=[]
    for s in selected:
        row=next(v for v in cache if v['row']['id']==s['id']);x0,y0,x1,y1=s['box']
        x=row['source'][None,:,y0:y1,x0:x1];t=row['target'][None,:,y0:y1,x0:x1]
        nodes=tr.net(x,all_nodes=True);p=nodes['rgb_head']/128
        saturation={k:dict(zero=float((v==0).float().mean()),high=float((v>=126.9).float().mean()),
                            std=float(v.std())) for k,v in nodes.items()}
        p=p[:,:,40:-40,40:-40];t=t[:,:,40:-40,40:-40]
        total,values=colour.loss_colour(p,t)
        adv=.08*(tr.critic(p)[-1]-1).square().mean()
        params=list(tr.net.parameters());vectors={}
        for key,loss in [('reconstruction',total),('boundary',1.2*values['boundary']),('adversarial',adv)]:
            gs=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
            vectors[key]=torch.cat([(torch.zeros_like(v) if g is None else g).flatten() for g,v in zip(gs,params)])
        norms={k:float(v.norm()) for k,v in vectors.items()}
        cosines={k:float(F.cosine_similarity(v[None],vectors['adversarial'][None])) for k,v in vectors.items() if k!='adversarial'}
        records.append(dict(id=s['id'],saturation=saturation,norms=norms,cosines_to_adversarial=cosines))
    write_json(live.OUT/'parameter_gradient_audit.json',records)
    print([{k:v for k,v in r.items() if k!='saturation'} for r in records],flush=True)
    print('SATURATION',[(r['id'],[(k,round(v['high'],3),round(v['zero'],3)) for k,v in r['saturation'].items() if v['high']>.2 or v['zero']>.8]) for r in records])


if __name__=='__main__':main()
