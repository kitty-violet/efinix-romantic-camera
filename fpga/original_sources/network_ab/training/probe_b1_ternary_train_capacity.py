"""Fixed TRAIN-only crop fit to separate optimization from generalization.

The graph, quantizer and current weighted style losses are unchanged. This is
a capacity diagnostic and cannot be delivered as a general-purpose style pack.
"""
import argparse,copy,json,random,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .train_b1_contour_coherence import ContourCoherenceTraining
from .train_b1_ternary_regions import TernaryB1,region_codes
from .train_b1_quantum_regions import RowScaledAdam
from .probe_b1_decoder_coverage import fragment
from . import train_b1_minimum_area as m
from .runtime import configure_workspace,file_hash,write_json
from .engine import set_reproducible


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,default=1000)
    a=ap.parse_args();configure_workspace();set_reproducible(2026100718);torch.set_num_threads(2)
    dest=Path(a.out);dest.mkdir(exist_ok=False);state,net=load(a.checkpoint);net=net.cuda();net.__class__=TernaryB1
    keys=[k for k in net.weight if k not in ('up1_high_dw','up2_low_dw','up2_low_pw','up2_high_dw','rgb_head','down1__dw','down2__dw')]
    groups=[]
    for k,w in net.weight.items():
        w.requires_grad_(k in keys);net.bias[k].requires_grad_(k in keys)
        if k in keys:
            s=(getattr(net,k+'_m')/131072).float().flatten()
            groups.extend([dict(params=[w],lr=.0003,row_gain=(s*256)[:,None,None,None]),dict(params=[net.bias[k]],lr=.002)])
    opt=RowScaledAdam(groups)
    levels=torch.cartesian_prod(torch.arange(15,device='cuda'),torch.arange(9,device='cuda'),torch.arange(9,device='cuda'));z=torch.zeros(1215,16,1,1,device='cuda')
    for c,(lo,hi) in enumerate(((0,7),(7,11),(11,15))):z[:,lo:hi,0,0]=(levels[:,c,None]-2*torch.arange(hi-lo,device='cuda')[None]).clamp(0,2)
    with torch.no_grad():palette=fragment(net,z,'up2_low_dw')['rgb_head'][:,:,0,0]/128
    pk=tuple(map(tuple,(palette.cpu().numpy()*128+128).tolist()));data=m.load_data(a.data)
    xs=[];ts=[];labs=[];codes=[];binding=[]
    for suffix,(xx,yy) in [('121509',(448,320)),('121831',(416,320))]:
        rows=[r for r in data['training'] if Path(r['video']).stem.endswith(suffix)];row=rows[len(rows)//2];assert row['split']=='train';sl=np.s_[yy:yy+320,xx:xx+320]
        def read(path):return torch.from_numpy(m.t.r.read(path)[sl].transpose(2,0,1).copy()).float()/128-1
        xs.append(read(row['path']));ts.append(read(row['target']));labs.append(torch.from_numpy(m.load_region(row['region'])[sl].astype(np.int64)));codes.append(torch.from_numpy(region_codes(row['target'],row['region'],pk)[sl].astype(np.int64)))
        binding.append(dict(id=row['id'],source=row['path'],crop=[xx,yy,320,320]))
    x,t=torch.stack(xs).cuda(),torch.stack(ts).cuda();tr=ContourCoherenceTraining.__new__(ContourCoherenceTraining)
    tr.net=net;tr.args=argparse.Namespace(classification=.5);tr.step=1;tr.rng=random.Random(4)
    tr.labels=torch.stack(labs).cuda();tr.class_labels=torch.stack(codes).cuda();tr.palette=palette;tr.pk=pk;tr.motion_aligned_fraction=x.new_zeros(())
    def update():
        opt.zero_grad(set_to_none=True);tr.nodes=net(x,all_nodes=True)
        loss,values=tr.additional_loss(tr.nodes['rgb_head'][...,40:-40,40:-40]/128,t[...,40:-40,40:-40],False)
        loss.backward();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);assert gn>0;opt.step()
        with torch.no_grad():
            for k in keys:
                lim=127*(getattr(net,k+'_m')/131072).float().flatten()[:,None,None,None]
                net.weight[k].copy_(net.weight[k].maximum(-lim).minimum(lim))
        return dict(loss=float(loss.detach()),**{k:float(v.detach()) for k,v in values.items()})
    original=copy.deepcopy(net.state_dict());update();saved=copy.deepcopy(net.state_dict());optim=copy.deepcopy(opt.state_dict());update();expected=copy.deepcopy(net.state_dict())
    net.load_state_dict(saved);opt.load_state_dict(optim);update();resume=max(float((v-expected[k]).abs().max()) for k,v in net.state_dict().items());assert resume==0
    net.load_state_dict(original);opt=RowScaledAdam(groups)
    write_json(dest/'binding.json',dict(checkpoint_sha256=file_hash(a.checkpoint),training_crops=binding,resume_error=resume,scope='Fixed training crops only; no generalization claim',code_sha256=file_hash(__file__)))
    print('TERNARY_CAPACITY_PREFLIGHT_PASS',resume,flush=True)
    @torch.no_grad()
    def review(step):
        nodes=net(x,all_nodes=True);integer=m.t.r.IntegerB1(m.t.r.runtime_from_image(net.export()),device='cuda');other,_=integer(x*128,True)
        err=max(float((v-other[k]).abs().max()) for k,v in nodes.items());assert err==0
        board=Image.new('RGB',(960,354*len(x)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
        for i in range(len(x)):
            for j,(title,img) in enumerate([('固定TRAIN源',x[i]),('绘画区域目标',t[i]),('实际整数B1',nodes['rgb_head'][i]/128)]):
                board.paste(m.t.r.c.rgb_image(img),(320*j,354*i+34));draw.text((320*j+4,354*i+5),title,font=font,fill='white')
        board.save(dest/f'fit_{step:06d}.png');return dict(step=step,all_node_error=err)
    history=[review(0)];start=time.perf_counter()
    with (dest/'metrics.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        for i in range(1,a.steps+1):
            values=update();log.write(json.dumps(dict(step=i,**values))+'\n')
            if i%100==0:print('TERNARY_CAPACITY',i,values['loss'],values['region_flatness'],values['ternary_count_ce'],flush=True)
            if i%250==0 or i==a.steps:history.append(dict(**review(i),values=values))
    state.update(model=net.state_dict(),step=a.steps,visual_acceptance=False,fixed_train_crop_diagnostic=True)
    torch.save(state,dest/'diagnostic.pt');write_json(dest/'review.json',dict(records=history,elapsed_seconds=time.perf_counter()-start,scope='No independent scenes or video; not deliverable'))


if __name__=='__main__':main()
