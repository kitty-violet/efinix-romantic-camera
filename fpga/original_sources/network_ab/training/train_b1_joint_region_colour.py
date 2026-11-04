"""Joint region geometry/colour learning with unrestricted RGB B1 coefficients.

Unlike the independent15-field count decoder, no fixed global colour lattice
is imposed. Region contrast is a vector constraint on neighbouring region
means; block interiors, contour sharpness and original target colour remain
separate reported criteria. The inference graph is unchanged plain B1.
"""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .train_b1_region_formation import formation_loss
from .train_b1_clean_regions import CleanRegionTraining
from .train_b1_native_coverage_counts import planned_native
from . import train_b1_late_region_supervision as driver


def joint_region_loss(p,t,labels):
    colours=[];flats=[];contrasts=[]
    boundary=torch.zeros_like(labels,dtype=torch.bool)
    boundary[:,:,1:]|=labels[:,:,1:]!=labels[:,:,:-1]
    boundary[:,1:]|=labels[:,1:]!=labels[:,:-1]
    valid=(F.max_pool2d(boundary[:,None].float(),5,1,2)==0)[:,0]
    for pp,tt,ll,ok in zip(p,t,labels,valid):
        # Compact crop IDs; absent global labels must not get equal weight.
        unique,index=torch.unique(ll.flatten(),sorted=True,return_inverse=True)
        n=len(unique);ids=index[None].expand(3,-1);weight=ok.flatten().float()
        count=weight.new_zeros(n).scatter_add_(0,index,weight)
        denom=count.clamp_min(1)
        mu=pp.new_zeros(3,n).scatter_add(1,ids,pp.flatten(1)*weight)/denom
        mt=tt.new_zeros(3,n).scatter_add(1,ids,tt.flatten(1)*weight)/denom
        area=count.sqrt();norm=area.sum().clamp_min(1)
        colours.append(((mu-mt).abs().mean(0)*area).sum()/norm)
        deviation=(pp.flatten(1)-mu[:,index]).abs().mean(0)*weight
        per_region=pp.new_zeros(n).scatter_add(0,index,deviation)/denom
        flats.append((per_region*area).sum()/norm)
        compact=index.reshape(ll.shape);keys=[]
        for a,b in ((compact[1:],compact[:-1]),(compact[:,1:],compact[:,:-1])):
            active=a!=b
            keys.append(torch.minimum(a[active],b[active])*n+torch.maximum(a[active],b[active]))
        pair,length=torch.unique(torch.cat(keys),return_counts=True)
        a,b=pair//n,pair%n
        wanted=mt[:,a]-mt[:,b];actual=mu[:,a]-mu[:,b]
        strong=(count[a]>=4)&(count[b]>=4)&(wanted.abs().amax(0)>=8/128)
        w=length.float().sqrt()*strong
        contrast=(F.smooth_l1_loss(actual,wanted,beta=8/128,reduction='none').mean(0)*w).sum()/w.sum().clamp_min(1)
        contrasts.append(contrast)
    colour=torch.stack(colours).mean();flat=torch.stack(flats).mean();contrast=torch.stack(contrasts).mean()
    _,shape=formation_loss(p,t,labels)
    pixel=F.l1_loss(p,t)
    # All targets are stylized region colours, not photographic reconstruction.
    total=colour+2.5*flat+contrast+.5*pixel+2*shape['missing_sharp_boundary']+4*shape['excess_diffuse_transitions']
    return total,dict(region_colour=colour,region_flatness=flat,adjacent_region_contrast=contrast,
        target_pixel_colour=pixel,sharp_boundary=shape['missing_sharp_boundary'],
        excess_transitions=shape['excess_diffuse_transitions'])


class JointRegionColourTraining(m.AreaTraining):
    preflight_native_resume=True
    input_stability_loss=CleanRegionTraining.input_stability_loss

    def __init__(self,args,data):
        assert not args.freeze_head and not args.class_manifest
        super().__init__(args,data)
        assert all(p.requires_grad for p in self.net.parameters())
        # Explicit bias step, rather than inheriting30x from reconstruction.
        self.opt.param_groups[1]['lr']=.002
        self.base_lrs=[g['lr'] for g in self.opt.param_groups]

    @staticmethod
    def appearance_loss(p,t,mode):return p.sum()*0,{}

    def additional_loss(self,p,t,native):
        labels=self.labels if native else self.labels[...,40:-40,40:-40]
        return joint_region_loss(p,t,labels)

    def update(self,native=None):
        return super().update(planned_native(self.step) if native is None else native)

    def preflight_extras(self):
        rows=[]
        while self.step%4:rows.append(self.update(True))
        row=self.update(False);assert row['temporal_pair'] and 'tiny_input_stability' in row['values']
        rows.append(row);return rows

    def save(self,path):
        super().save(path);state=torch.load(path,map_location='cpu',weights_only=True)
        state.update(joint_region_colour=True,fixed_colour_palette=False,
            inference_added_operators=0,minimum_area_inference_guarantee=False,visual_acceptance=False,
            acceptance='All criteria plus actual integer video required; scalar score is diagnostic only')
        torch.save(state,path)


@torch.inference_mode()
def evaluate(net,rows):
    records=[]
    for row in rows:
        x=m.t.r.tensor(m.t.r.read(row['path']))[None];target=m.t.r.tensor(m.t.r.read(row['target']))[None]
        p=net(x)[0];labels=torch.tensor(m.load_region(row['region']).astype(np.int64),device=x.device)[None]
        total,values=joint_region_loss(p,target,labels)
        # Report signed chroma drift separately: averaging all RGB errors can
        # hide purple/green casts in otherwise low-error large backgrounds.
        delta=(p-target)*128;r,g,b=delta.unbind(1)
        records.append(dict(id=row['id'],**m.t.r.metrics(p,target),**{k:float(v) for k,v in values.items()},
            bias_Co=float(((r-b)/2).mean()),bias_Cg=float(((-r+2*g-b)/4).mean()),score=float(total)))
    return dict(records=records,mean={k:sum(r[k] for r in records)/len(records) for k in records[0] if k!='id'})


@torch.inference_mode()
def preview(net,rows,dest):
    dest.mkdir(exist_ok=True);images={}
    for row in rows:
        source=m.t.r.read(row['path']);p=net(m.t.r.tensor(source)[None])[0][0]
        output=m.t.r.c.rgb_image(p);output.save(dest/(row['id']+'_output.png'))
        images[row['id']]=[Image.fromarray(source),Image.fromarray(m.t.r.read(row['target'])),output]
    board=Image.new('RGB',(1260,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        for j,(im,title) in enumerate(zip(images[key],('原图','修订区域目标', '整数B1输出' if net.quantized else 'FP32网络输出·非部署验收'))):
            board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+4,314*i+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'comparison.png')


def regression():
    labels=torch.zeros(1,96,128,dtype=torch.long);labels[:,:,48:80]=1;labels[:,:,80:]=2
    palette=torch.tensor([[-.4,-.25,-.2],[.1,.3,.45],[.65,.55,.45]])
    target=palette[labels].permute(0,3,1,2).contiguous()
    blurred=F.avg_pool2d(F.pad(target,(6,6,6,6),mode='replicate'),13,1)
    merged=target.clone();merged[:,:,:,48:80]=palette[0][None,:,None,None]
    cast=target+torch.tensor([.1,-.1,.1])[None,:,None,None]
    noise=target.clone();noise[:,:,20:24,20:24]+=.3
    shifted=target.roll(2,-1);shifted[:,:,:,:2]=target[:,:,:,:2]
    cases=dict(exact=target,blurred=blurred,merged_layers=merged,purple_cast=cast,speck=noise,shift2=shifted)
    values={k:{'total':float(joint_region_loss(p,target,labels)[0]),
        **{key:float(v) for key,v in joint_region_loss(p,target,labels)[1].items()}} for k,p in cases.items()}
    assert values['exact']['total']<1e-4
    assert values['merged_layers']['adjacent_region_contrast']>.05
    assert values['purple_cast']['region_colour']>.09
    assert values['speck']['region_flatness']>0
    assert values['blurred']['total']>values['shift2']['total']
    flat=torch.zeros_like(target,requires_grad=True);loss,_=joint_region_loss(flat,target,labels);loss.backward()
    assert flat.grad[:,:,:,:40].mean()>0 and flat.grad[:,:,:,88:].mean()<0
    print('JOINT_REGION_LOSS_REGRESSION_PASS',values,flush=True)
    return values


if __name__=='__main__':
    regression()
    driver.main(training_factory=JointRegionColourTraining,extra_sources=[Path(__file__)],evaluation=evaluate,preview=preview)
