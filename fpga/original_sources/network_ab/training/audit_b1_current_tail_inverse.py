"""Known-target latent optimization through the CURRENT frozen B1 tail.

This is a decoder-capacity diagnostic, NEVER a source prediction or a usable
weight package. Only signed-byte res3_add feature values change, not weights.
It differs from the historical14-luma-label inversion of a fixed RGB palette.
"""
import argparse,json,shutil
from pathlib import Path
import torch,numpy as np
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .b1_geometry_first_objective import geometry_first
from .audit_b1_rim_tolerance import rim_terms
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--mining',required=True);ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,default=800);a=ap.parse_args()
    torch.set_num_threads(2);torch.manual_seed(793);_,net=load(a.checkpoint);net.requires_grad_(False)
    rows={r['id']:r for r in m.load_data(a.data)['training']};items=json.loads(Path(a.mining).read_text(encoding='utf-8'))['records'];xs=[];ts=[];labs=[];ids=[]
    for item in (items[1],items[5],items[6]):
        row=rows[item['row']['id']];xx,yy=item['xy'];w,h=item['wh']
        tensor=lambda key:torch.from_numpy(m.t.r.read(row[key])[yy:yy+h,xx:xx+w].transpose(2,0,1).copy()).float()/128-1
        xs.append(tensor('path'));ts.append(tensor('target'));labs.append(torch.from_numpy(m.load_region(row['region'])[yy:yy+h,xx:xx+w].astype(np.int64)));ids.append(row['id'])
    x=torch.stack(xs);target=torch.stack(ts);labels=torch.stack(labs);cut=lambda z:z[...,40:-40,40:-40]
    with torch.no_grad():
        full=net(x,all_nodes=True);baseline=full['rgb_head']/128;z0=full['res3_add']
        replay=fragment(net,z0,'res3_add');assert all(torch.equal(full[k],v) for k,v in replay.items() if k not in ('res1_add','res2_add'))
        actual,_=m.t.r.IntegerB1(m.t.r.runtime_from_image(net.export()),device='cpu')(net.round(x*128),True)
        assert all(torch.equal(v,actual[k]) for k,v in full.items())
    raw=net.export();latent=torch.nn.Parameter(z0.clone());optimizer=torch.optim.Adam([latent],lr=.5)
    dest=Path(a.out);dest.mkdir(exist_ok=False);shutil.copy2(__file__,dest/Path(__file__).name);history=[]
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
    def save(step,p,loss):
        board=Image.new('RGB',(960,1044),'#17202a');draw=ImageDraw.Draw(board)
        for j in range(3):
            for k,(name,v) in enumerate((('固定目标',target[j]),('当前源预测',baseline[j]),('仅诊断：已知目标优化32场',p[j]))):
                board.paste(m.t.r.c.rgb_image(v),(320*k,348*j+28));draw.text((320*k+3,348*j+3),name,font=font,fill='white')
        board.save(dest/f'diagnostic_{step:06d}.png')
        row=dict(step=step,loss=float(loss),mae=float((cut(p)-cut(target)).abs().mean()*128),feature_change_mae=float((net.round(latent)-z0).abs().mean()))
        history.append(row);write_json(dest/'history.json',history);print('CURRENT_TAIL_INVERSE',row,flush=True)
    for step in range(a.steps+1):
        z=net.round(latent).clamp(-128,127);p=fragment(net,z,'res3_add')['rgb_head']/128
        loss=geometry_first(cut(p),cut(target),cut(labels))[0]+.75*rim_terms(cut(p),cut(target),cut(labels),2)[0]+(cut(p)-cut(target)).abs().mean()
        if step%200==0 or step==a.steps:save(step,p.detach(),loss.detach())
        if step==a.steps:break
        optimizer.zero_grad();loss.backward();assert torch.isfinite(latent.grad).all() and latent.grad.norm()>0;optimizer.step()
        with torch.no_grad():latent.clamp_(-128,127)
    assert net.export()==raw
    torch.save(dict(features=net.round(latent.detach()).to(torch.int8),ids=ids,node='res3_add',uses_target_at_optimization=True),dest/'diagnostic_latent.pt')
    write_json(dest/'results.json',dict(history=history,checkpoint_sha256=file_hash(a.checkpoint),training_ids=ids,bytes_unchanged=True,
        initial_full_graph_and_all_nodes_error=0,only_decoder_diagnostic=True,not_source_inference=True,not_a_weight_candidate=True,
        limitation='Finite optimization, not proven global optimum or architecture capacity bound.'))


if __name__=='__main__':main()
