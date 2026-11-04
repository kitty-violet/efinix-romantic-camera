"""Exact existing package plus a NEW documented TRAIN optimizer.

No old optimizer continuation is claimed. Restore residual alignment from
the published byte package and require the legacy checkpoint to re-export
exactly those bytes before using any newly computed statistics.
"""
import argparse,copy,shutil
from pathlib import Path
import torch
from .b1_trained_image import decode
from .whitened_b1_optimizer import patch_statistics,WhitenedB1Adam
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','package','data','out'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();torch.set_num_threads(2);out=Path(a.out);out.mkdir(exist_ok=False)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    net=m.t.r.c.ColourB1(revive=False);net.load_state_dict(state['model']);net.quantized=True
    raw=Path(a.package).read_bytes();_,residual=decode(raw);net.package['residual_encoding']=residual
    assert net.export()==raw,'Legacy checkpoint differs from its published package'
    crops=[];ids=[];data=m.load_data(a.data)
    for video in sorted({r['video'] for r in data['training']}):
        rr=[r for r in data['training'] if r['video']==video];r=rr[len(rr)//2]
        rgb=m.t.r.read(r['path'])[200:520,480:800]
        crops.append(torch.from_numpy(rgb.transpose(2,0,1).copy()).float()/128-1);ids.append(r['id'])
    x=torch.stack(crops);stats=patch_statistics(net,x,4096)
    groups=[]
    for k,w in net.weight.items():
        s=(getattr(net,k+'_m')/131072).float().flatten()
        groups.extend([dict(params=[w],lr=.0002,row_gain=(s*256)[:,None,None,None]),dict(params=[net.bias[k]],lr=.004)])
    opt=WhitenedB1Adam(groups,net,stats,True);assert net.export()==raw
    with torch.no_grad():
        nodes=net(x[:1],all_nodes=True);actual,_=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')(net.round(x[:1]*128),True)
        assert all(torch.equal(v,actual[k]) for k,v in nodes.items())
    kind='Calibrated final720 clip-code package' if 'final720_clip_design' in state else 'Old identity reconstruction package'
    torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),step=0,residual_encoding=copy.deepcopy(residual),
        contract_sha256=state['contract_sha256'],qat=True,visual_acceptance=False,
        initialization=kind+', exact bytes, new TRAIN whitening optimizer',training_ids=ids),out/'initial.pt')
    shutil.copy2(__file__,out/Path(__file__).name)
    write_json(out/'review.json',dict(args=vars(a),checkpoint_sha256=file_hash(a.checkpoint),package_sha256=file_hash(a.package),
        initialization_bytes_unchanged=True,actual_all_nodes_error=0,training_ids=ids,optimizer_reset=True,visual_acceptance=False))
    print('RECONSTRUCTION_SEED_PASS exact_old_bytes all_nodes0 new_optimizer',flush=True)


if __name__=='__main__':main()
