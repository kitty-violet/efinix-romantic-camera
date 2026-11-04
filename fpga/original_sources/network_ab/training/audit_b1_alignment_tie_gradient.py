"""Reproduce loss-alignment tie smearing; frozen historical implementation.

This is a gradient mechanism audit, not evidence that every soft edge has
this cause, nor a claim that weights are already visually repaired.
"""
import argparse,importlib.util,json,shutil
from pathlib import Path
import numpy as np,torch
from .b1_coherent_rgb_patch_profiles import coherent_profiles
from .probe_b1_encoder_gauge import load
from .runtime import file_hash,write_json
from . import train_b1_minimum_area as m


def compare(p,t,lab,old):
    result={};grad=[]
    for name,fn in [('historical_split_ties',old),('one_nearest_correspondence',coherent_profiles)]:
        z=p.detach().clone().requires_grad_();loss,_=fn(z,t,lab);g=torch.autograd.grad(loss,z)[0]
        result[name]=dict(loss=float(loss),gradient_l1=float(g.abs().sum()),
            target_direction=float((g*(t-p)).sum()),gradient_nonzero_pixels=int((g.abs().amax(1)>1e-10).sum()))
        grad.append(g)
    assert abs(result['historical_split_ties']['loss']-result['one_nearest_correspondence']['loss'])<1e-6
    result['gradient_max_difference']=float((grad[1]-grad[0]).abs().max())
    return result,grad


def main():
    ap=argparse.ArgumentParser()
    for k in ('historical','checkpoint','data','mining','out'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();torch.set_num_threads(2);out=Path(a.out);out.mkdir(exist_ok=False)
    spec=importlib.util.spec_from_file_location('network_ab.training._historical_coherent_profile',a.historical)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    results=[]
    for width in (1,2,3,4):
        lab=torch.zeros(1,72,96,dtype=torch.long);lab[...,48:48+width]=1
        target=lab[:,None].float().expand(-1,3,-1,-1)*.5;p=target*0
        row,gg=compare(p,target,lab,mod.coherent_profiles)
        for key,g in zip(('historical_split_ties','one_nearest_correspondence'),gg):
            row[key]['profile_gradient_abs_mass']=g.abs().sum((0,1,2))[40:60].tolist()
            row[key]['in_foreground_abs_mass']=float((g.abs()*lab[:,None]).sum())
        assert row['gradient_max_difference']>0
        assert row['one_nearest_correspondence']['target_direction']<row['historical_split_ties']['target_direction']
        results.append(dict(width=width,**row))
    data=m.load_data(a.data);byid={r['id']:r for r in data['training']}
    mined=json.loads(Path(a.mining).read_text('utf-8'))['records'];_,net=load(a.checkpoint);net.requires_grad_(False)
    natural=[]
    for i in (1,5,6):
        s=mined[i];row=byid[s['row']['id']];xx,yy=s['xy'];w,h=s['wh']
        ten=lambda k:torch.from_numpy(m.t.r.read(row[k])[yy:yy+h,xx:xx+w].transpose(2,0,1).copy())[None].float()/128-1
        p=net(ten('path'))[0][...,40:-40,40:-40];t=ten('target')[...,40:-40,40:-40]
        lab=torch.from_numpy(m.load_region(row['region'])[yy+40:yy+h-40,xx+40:xx+w-40].astype(np.int64))[None]
        v,_=compare(p,t,lab,mod.coherent_profiles);natural.append(dict(id=row['id'],**v))
    shutil.copy2(__file__,out/Path(__file__).name)
    write_json(out/'results.json',dict(synthetic=results,natural=natural,historical_sha256=file_hash(a.historical),
        corrected_sha256=file_hash(Path(__file__).with_name('b1_coherent_rgb_patch_profiles.py')),
        same_forward_loss=True,not_a_visual_acceptance=True,scope='Exact-tie backward selection only. One contributing mechanism, not all-blur attribution.'))
    print('ALIGNMENT_TIE_AUDIT',[(r['width'],r['historical_split_ties']['in_foreground_abs_mass'],r['one_nearest_correspondence']['in_foreground_abs_mass']) for r in results],natural,flush=True)


if __name__=='__main__':main()
