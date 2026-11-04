"""Seed seven local contour/colour fields from existing reconstruction features."""
import argparse
import shutil
from pathlib import Path
import torch
from . import train_b1_minimum_area as m
from .probe_b1_two_colour_geometry import install,latent
from .publish_b1_style_recovery import export
from .runtime import configure_workspace,write_json,file_hash


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--name',required=True);a=ap.parse_args()
    configure_workspace();torch.set_num_threads(2);dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    state=torch.load(a.warm,map_location='cpu',weights_only=True);net=m.t.r.initialize();net.load_state_dict(state['model']);net.quantized=True
    if 'residual_encoding' in state:
        net.package['residual_encoding']=state['residual_encoding']
    else:
        # Older reconstruction checkpoints keep these three records only in
        # their published coefficient image. Verify that it is the same model.
        from .b1_trained_image import decode
        raw=(Path(a.warm).parent/'coefficients-padded.bin').read_bytes()
        _,net.package['residual_encoding']=decode(raw)
        assert net.export()==raw,'Warm checkpoint and published image differ'
    install(net);d=m.load_data();rows=d['training'][::4];xs=[];ys=[]
    for i,row in enumerate(rows):
        f=net(m.t.r.tensor(m.t.r.read(row['path']))[None],all_nodes=True)['up1_low_dw']
        truth=torch.from_numpy(latent(row['target'],row['region'])).cuda()
        xs.append(f[0,:,::3,::3].flatten(1).T);ys.append(truth[:,::3,::3].flatten(1).T)
        if (i+1)%10==0:print('TWO_COLOUR_FEATURES',i+1,len(rows),flush=True)
    x=torch.cat(xs).double();y=torch.cat(ys).double();mu=x.mean(0);sd=x.std(0).clamp_min(2);xn=(x-mu)/sd
    design=torch.cat((xn,torch.ones(len(x),1,device='cuda',dtype=torch.float64)),1)
    ridge=torch.eye(33,device='cuda',dtype=torch.float64)*len(x)*.001;ridge[-1,-1]=0
    theta=torch.linalg.solve(design.T@design+ridge,design.T@y);w=(theta[:32].T/sd[None]).float();b=(theta[-1]-mu@(theta[:32]/sd[:,None])).float()
    k='up1_low_pw';net.weight[k].zero_();net.bias[k].zero_();net.weight[k][:7,:,0,0]=w;net.bias[k][:7]=b
    scale=net.weight[k].abs().flatten(1).amax(1).clamp_min(1e-5)/120
    getattr(net,k+'_m').copy_((scale*131072).ceil().clamp(1,131071).double()[None,:,None,None])
    export(net,d['validation'],dest/'package');m.t.preview(net,d['validation'],dest/'preview')
    torch.save(dict(model=net.state_dict(),qat=True,step=0,residual_encoding=net.package['residual_encoding'],
        two_colour_fields=True,parent=a.warm,coverage_layout='two_rgb_sides_and_signed_contour'),dest/'initial.pt')
    for p in (Path(__file__),Path(__file__).with_name('probe_b1_two_colour_geometry.py')):shutil.copy2(p,dest/p.name)
    write_json(dest/'review.json',dict(training_ids=[r['id'] for r in rows],data_sha256=file_hash(m.OUT/'data.json'),
        samples=len(x),field_fit_mae=float((design@theta-y).abs().mean()),added_operators=0,postfilter=False,visual_acceptance=False))
    print('TWO_COLOUR_INITIAL_COMPLETE',flush=True)


if __name__=='__main__':main()
