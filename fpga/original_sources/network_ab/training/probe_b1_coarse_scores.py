"""Broaden colour-decision support, never blur an already coloured RGB image."""
import argparse
import copy
import shutil
from pathlib import Path
import torch
from . import train_b1_minimum_area as m
from .publish_b1_style_recovery import export
from .runtime import configure_workspace,write_json


@torch.no_grad()
def coarse(net,amount):
    for key in ('res1_conv1__dw','res1_conv2__dw'):
        w=net.weight[key]
        w.copy_((1-amount)*w+amount*w.sum((-2,-1),keepdim=True)/9)
        s=torch.pow(2,torch.ceil(torch.log2(w.abs().flatten(1).amax(1).clamp_min(1e-5)/127)))
        getattr(net,key+'_m').copy_((s*131072).clamp(1,131071).double()[None,:,None,None])
    key='res1_conv2__dw';s=(getattr(net,key+'_m')/131072).float().flatten()[:16]
    q=net.round(net.weight[key][:16]/s[:,None,None,None]).clamp(-127,127)
    assert q.min()>=0 and net.round(q.flatten(1).sum(1)*s*127).max()<=7


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--name',required=True);a=ap.parse_args()
    configure_workspace();torch.set_num_threads(2);dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    state=torch.load(a.warm,map_location='cpu',weights_only=True);net=m.t.r.initialize();net.load_state_dict(state['model'])
    net.package['residual_encoding']=state['residual_encoding'];net.quantized=True;original=copy.deepcopy(net.state_dict());data=m.load_data()
    for name,amount in [('half',.5),('full',1.)]:
        net.load_state_dict(original);coarse(net,amount);path=dest/name;path.mkdir()
        export(net,data['validation'],path/'package');m.t.preview(net,data['validation'],path/'preview')
        saved=dict(state);saved.update(model=net.state_dict(),coarse_decision_support=amount,parent=a.warm)
        torch.save(saved,path/'initial.pt')
    shutil.copy2(__file__,dest/Path(__file__).name)
    write_json(dest/'review.json',dict(parent=a.warm,only_two_preselection_depthwise_layers_changed=True,
        rgb_decoder_unchanged=True,postfilter=False,visual_acceptance=False))
    print('COARSE_SCORE_PROBE_COMPLETE',flush=True)


if __name__=='__main__':main()
