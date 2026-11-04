"""Ablate mask geometry inside B1, never filter the rendered RGB image."""
import argparse
import copy
import shutil
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageFont
from . import train_b1_minimum_area as m
from .b1_trained_image import runtime_from_image
from .probe_b1_cartoon_int8 import IntegerB1
from .runtime import configure_workspace, write_json, file_hash
from .engine import set_reproducible


@torch.no_grad()
def geometry(net, smoothing, gain, aa=8):
    """Smooth region coverage, clamp to flat interiors, then mix once."""
    kernel=net.weight['stem'].new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
    key='up2_low_dw';net.weight[key].zero_()
    if smoothing:net.weight[key][:,0]=gain*kernel
    else:net.weight[key][:,0,1,1]=gain
    net.bias[key].fill_(-(gain-1)*63.5)
    old_aa=round(float(net.weight['up2_high_dw'][0,0].sum()*127))
    net.weight['up2_high_dw'][:,0]=kernel*aa/127
    net.weight['rgb_head']*=old_aa/aa
    for key in ('up2_low_dw','up2_high_dw','rgb_head'):
        w=net.weight[key]
        s=torch.pow(2,torch.ceil(torch.log2(w.abs().flatten(1).amax(1).clamp_min(1e-5)/127)))
        getattr(net,key+'_m').copy_((s*131072).clamp(1,131071).double()[None,:,None,None])


@torch.no_grad()
def soften_threshold(net,gain):
    assert net.package['residual_encoding']['res3_add']['main_scaled'] is False
    levels=[(0,q) for q in (14,28,42,56,70,84,98,112)]
    levels += [(4,q) for q in (49,59,69,79)]+[(8,q) for q in (49,59,69,79)]
    key='res3_conv1__pw';net.weight[key].zero_();net.bias[key].zero_()
    for c,(j,q) in enumerate(levels):
        net.weight[key][c,j,0,0]=gain;net.bias[key][c]=63.5-gain*(q-.5)
    w=net.weight[key];s=torch.pow(2,torch.ceil(torch.log2(w.abs().flatten(1).amax(1).clamp_min(1e-5)/127)))
    getattr(net,key+'_m').copy_((s*131072).clamp(1,131071).double()[None,:,None,None])


@torch.no_grad()
def binomial_sampling(net):
    """Binomial shape with exactly preserved integer DC gain for each row."""
    kernel=net.weight['stem'].new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
    for key in ('down1__dw','down2__dw'):
        scale=(getattr(net,key+'_m')/131072).float().flatten()
        old=net.round(net.weight[key][:,0]/scale[:,None,None]).clamp(-127,127)
        dc=old.sum((1,2));assert (dc>0).all()
        taps=net.round(dc[:,None,None]*kernel)
        taps[:,1,1]+=dc-taps.sum((1,2))
        assert (taps>=0).all() and (taps<=127).all() and torch.equal(taps.sum((1,2)),dc)
        net.weight[key][:,0]=taps*scale[:,None,None]
        # For every constant integer input, dot product/bias/requantization
        # therefore remain identical, including replicated image boundaries.


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--name',required=True)
    ap.add_argument('--soft-thresholds',action='store_true');ap.add_argument('--sampling',action='store_true')
    ap.add_argument('--crisp',action='store_true');ap.add_argument('--morphology',action='store_true');a=ap.parse_args()
    assert sum((a.soft_thresholds,a.sampling,a.crisp,a.morphology))<=1
    configure_workspace();set_reproducible(20261006161);torch.set_num_threads(2)
    dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    state=torch.load(a.warm,map_location='cpu',weights_only=True);net=m.t.r.initialize();net.load_state_dict(state['model']);net.quantized=True
    net.package['residual_encoding']=state['residual_encoding'];initial=copy.deepcopy(net.state_dict());d=m.load_data()
    images={r['id']:[Image.fromarray(m.t.r.read(r['target']))] for r in d['validation']}
    trials=[('current',False,3.,4),('aa8',False,3.,8),('shape2',True,2.,8),('shape3',True,3.,8),('shape4',True,4.,8)]
    if a.soft_thresholds:trials=[('current',False,3.,4)]+[(f'soft{g}',True,3.,8) for g in (8,16,32,64)]
    if a.sampling:trials=[('current',False,3.,4),('sampling',False,3.,4),('sampling-aa8',False,3.,8),('sampling-soft16',True,3.,8),('sampling-shape3',True,3.,8)]
    if a.crisp:trials=[('current',True,3.,8),('gain6',True,6.,8),('gain6-light-aa',True,6.,8),('gain8-light-aa',True,8.,8),('gain8-point',True,8.,8)]
    if a.morphology:trials=[(name,True,6.,8) for name in ('current','no-morph','majority3','majority3x2','majority3x2-single-aa')]
    for name,smooth,gain,aa in trials:
        net.load_state_dict(initial)
        if name!='current' and not a.morphology:geometry(net,smooth,gain,aa)
        if a.morphology:
            from .probe_b1_decoder_coverage import geometry_variant
            geometry_variant(net,name)
        if name.startswith('soft'):soften_threshold(net,int(name[4:]))
        if name.startswith('sampling'):binomial_sampling(net)
        if name=='sampling-soft16':soften_threshold(net,16)
        if 'light-aa' in name or name.endswith('-point'):
            alpha=.25 if 'light-aa' in name else 0.
            kernel=net.weight['stem'].new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16*alpha
            kernel[1,1]+=1-alpha;key='up2_high_dw';net.weight[key][:,0]=kernel*aa/127
            w=net.weight[key];s=torch.pow(2,torch.ceil(torch.log2(w.abs().flatten(1).amax(1)/127)))
            getattr(net,key+'_m').copy_((s*131072).clamp(1,131071).double()[None,:,None,None])
        out=dest/name;out.mkdir();raw=net.export();(out/'coefficients-padded.bin').write_bytes(raw)
        actual=IntegerB1(runtime_from_image(raw))
        # Quick node comparison on a small image. Finalists still need full
        # six-frame and independent scalar-oracle verification before delivery.
        sample=m.t.r.tensor(m.t.r.read(d['validation'][0]['path']))[None,...,:64,:64]
        ours=net(sample,all_nodes=True);got,_=actual(sample*128,True)
        err=max(float((v-got[k]).abs().max()) for k,v in ours.items());assert err==0
        del ours,got
        for row in d['validation']:
            x=m.t.r.tensor(m.t.r.read(row['path']))[None]
            q=actual(x*128)/128;image=m.t.r.c.rgb_image(q[0]);image.save(out/(row['id']+'.png'));images[row['id']].append(image)
        torch.save(dict(model=net.state_dict(),qat=True,step=state.get('step'),residual_encoding=state['residual_encoding'],
                        field_node=state.get('field_node','res2_add'),parent=a.warm,
                        fixed_sampling=state.get('fixed_sampling',False) or name.startswith('sampling'),
                        classifier_mode=state.get('classifier_mode',False)),out/'initial.pt')
        write_json(out/'probe.json',dict(geometry=dict(smoothing=smooth,gain=gain,aa=aa),small_node_error=err,
                                       visual_acceptance=False,full_export_verification_pending=True))
        print('COVERAGE_PROBE',name,flush=True)
        del actual;torch.cuda.empty_cache()
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15)
    for first in (1,3):
        names=[v[0] for v in trials[first-1:first+2]]
        board=Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
        columns=[0,first,first+1,first+2]
        for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
            for j,column in enumerate(columns):
                board.paste(images[key][column].crop(box),(j*420,i*314+34));title=(['离线目标']+names)[j]
                draw.text((j*420+4,i*314+5),label+' | '+title,font=font,fill='white')
        board.save(dest/f'comparison_{first}.png')
    shutil.copy2(__file__,dest/Path(__file__).name)
    if a.morphology:shutil.copy2(Path(__file__).with_name('probe_b1_decoder_coverage.py'),dest/'probe_b1_decoder_coverage.py')
    write_json(dest/'review.json',dict(parent=a.warm,parent_sha256=file_hash(a.warm),code_sha256=file_hash(__file__),
        real_byte_inference=True,postfilter=False,added_operators=0,visual_acceptance=False))


if __name__=='__main__':main()
