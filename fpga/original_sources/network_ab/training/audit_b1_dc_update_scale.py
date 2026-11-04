"""Replay one saved WhitenedB1Adam step, varying only its centred-bias delta.

Actual integer inference is measured; this does not establish a learning-rate
recommendation or visual success from a single favourable training batch.
"""
import argparse,copy,json,shutil
from pathlib import Path
import numpy as np,torch
from .probe_b1_encoder_gauge import load
from .train_b1_trainable_luma_readout import ReadoutB1
from .whitened_b1_optimizer import WhitenedB1Adam,patch_statistics
from .train_b1_plain_blocks_native import objective
from .b1_coherent_rgb_patch_profiles import coherent_profiles
from .runtime import file_hash,write_json
from . import train_b1_minimum_area as m


def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','mining','out'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();torch.set_num_threads(2);state,net=load(a.checkpoint);net.__class__=ReadoutB1
    rows={r['id']:r for r in m.load_data(a.data)['training']}
    specs=json.loads(Path(a.mining).read_text(encoding='utf-8'))['records'];specs=[specs[k] for k in (1,5,6)]
    xs=[];ts=[];ls=[]
    for s in specs:
        row=rows[s['row']['id']];x,y=s['xy'];w,h=s['wh']
        for key,dst in (('path',xs),('target',ts)):
            dst.append(torch.from_numpy(m.t.r.read(row[key])[y:y+h,x:x+w].transpose(2,0,1).copy()).float()/128-1)
        ls.append(torch.from_numpy(m.load_region(row['region'])[y:y+h,x:x+w].astype(np.int64)))
    x,t,lab=torch.stack(xs),torch.stack(ts),torch.stack(ls);cut=lambda z:z[...,40:-40,40:-40]
    groups=[];stats=patch_statistics(net,x,512)
    for k,w in net.weight.items():
        s=(getattr(net,k+'_m')/131072).float().flatten()
        groups.extend([dict(params=[w],lr=.0002,row_gain=(s*256)[:,None,None,None]),dict(params=[net.bias[k]],lr=.004)])
    opt=WhitenedB1Adam(groups,net,stats,True);opt.load_state_dict(state['optimizer'])
    assert [g.get('node_key') for g in opt.param_groups[::2]]==list(net.weight)
    start=copy.deepcopy(net.state_dict());p=net(x)[0]
    loss,_=objective(cut(p),cut(t));loss.backward();opt.prepare_gradients()
    torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);opt.step()
    after=copy.deepcopy(net.state_dict());dc={};layers=[]
    with torch.no_grad():
        for group in opt.param_groups:
            if 'node_key' not in group:continue
            k=group['node_key'];s=(getattr(net,k+'_m')/131072).float().flatten()[:,None,None,None]
            dw=(net.round(net.weight[k]/s)-net.round(start['weight.'+k]/s))*s
            db=net.bias[k]-start['bias.'+k]
            dc[k]=db+(dw.flatten(1)*group['patch_mean']).sum(1)
            layers.append(dict(layer=k,dc_step_max=float(dc[k].abs().max()),dc_step_median=float(dc[k].abs().median())))
        records=[]
        for factor in (0,1,4,16,64):
            net.load_state_dict(start if factor==0 else after)
            if factor:
                for k in dc:net.bias[k].add_(dc[k]*(factor-1))
            p=net(x)[0];score,_=objective(cut(p),cut(t));shape,_=coherent_profiles(cut(p),cut(t),cut(lab))
            noisy=net((x+1/128).clamp(-1,127/128))[0]
            records.append(dict(factor=factor,score=float(score),mae=float((cut(p)-cut(t)).abs().mean()*128),
                coherent=float(shape),one_byte_input_response=float((cut(p)-cut(noisy)).abs().mean()*128)))
    out=Path(a.out);out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name)
    write_json(out/'review.json',dict(records=records,layers=layers,checkpoint_sha256=file_hash(a.checkpoint),
        source_sha256=file_hash(__file__),scope='one saved-moment update on3TRAIN; not a recommended LR or acceptance',
        objective_excludes_shift_flow_noise=True))
    print('DC_UPDATE_AUDIT',records,flush=True)


if __name__=='__main__':main()
