"""Re-encode existing final-DW gain, keeping the63.5 class threshold.

No added comparator: saturation is the existing B1 activation. Larger logit
temperature during the next fit separates classification confidence from
the RGB mixing interval. Only original Q17 coefficients are exported.
"""
import argparse,copy,shutil
from pathlib import Path
import torch
from .probe_b1_encoder_gauge import load
from .audit_b1_structural_task_updates import restored_optimizer
from .runtime import file_hash,write_json
from . import train_b1_minimum_area as m


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','out'):ap.add_argument('--'+k,required=True)
    ap.add_argument('--gain',type=int,default=8)
    a=ap.parse_args();torch.set_num_threads(2);out=Path(a.out);out.mkdir(exist_ok=False)
    state,net=load(a.checkpoint);before=copy.deepcopy(net);key='up2_high_dw';gain=a.gain
    assert gain>=1 and (net.up2_high_dw_m*gain<=131071).all()
    net.weight[key].mul_(gain);net.bias[key].copy_(gain*(net.bias[key]-63.5)+63.5);net.up2_high_dw_m.mul_(gain)
    opt=restored_optimizer(net,state['optimizer']);opt.state.clear()
    for group in opt.param_groups:
        if group['params'][0] is net.weight[key]:group['row_gain']*=gain
        if group['params'][0] is net.bias[key]:group['lr']*=gain
    raw=net.export();assert len(raw)==20480;(out/'coefficients-padded.bin').write_bytes(raw)
    rt=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu');records=[]
    rows=m.load_data(a.data)['training'];chosen=[]
    for video in sorted({r['video'] for r in rows}):
        rr=[r for r in rows if r['video']==video];chosen.append(rr[len(rr)//2])
    for row in chosen:
        src=m.t.r.read(row['path'])[200:520,480:800]
        x=torch.from_numpy(src.transpose(2,0,1).copy())[None].float()/128-1
        p=before(x,all_nodes=True);q=net(x,all_nodes=True);actual,_=rt(net.round(x*128),True)
        assert all(torch.equal(v,actual[k]) for k,v in q.items())
        same=(p[key]>=64)==(q[key]>=64);different=float((~same).float().mean())
        assert different<.002,(row['id'],different)
        rec=dict(id=row['id'],class_change_fraction=different,all_nodes_error=0,
                 old_mid=float(((p[key]>12)&(p[key]<115)).float().mean()),
                 new_mid=float(((q[key]>12)&(q[key]<115)).float().mean()))
        records.append(rec);print('MARGIN_RECODE',rec,flush=True)
    state.update(model=net.state_dict(),optimizer=opt.state_dict(),step=0,visual_acceptance=False,
        classifier_margin_recode=dict(gain=gain,threshold=63.5,parent=a.checkpoint,
            recommended_code_temperature=16*gain,optimizer_reset=True,no_inference_operator=True))
    torch.save(state,out/'initial.pt');shutil.copy2(__file__,out/Path(__file__).name)
    write_json(out/'review.json',dict(args=vars(a),records=records,checkpoint_sha256=file_hash(a.checkpoint),
        package_sha256=file_hash(out/'coefficients-padded.bin'),visual_acceptance=False,
        same_class_up_to_measured_requantization=True,extra_inference_ops=0))


if __name__=='__main__':main()
