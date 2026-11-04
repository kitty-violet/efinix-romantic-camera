"""Teach texture rejection inside known colour regions, within original B1.

Procedural region labels and paired colour augmentation are training only.
Natural high-frequency textures come exclusively from the existing train split.
No segmentation, filtering, palette lookup or temporal state is added to export.
"""
from pathlib import Path
import json
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from .train_b1_balanced_region_boundaries import BalancedBoundaryTraining
from .train_b1_ordered_regions import evaluate,preview as natural_preview
from .build_b1_ordered_palette import actual_palette
from .train_b1_sixteen_regions import colour_target
from .minimum_area_blocks import area_merge,components,boundary_mode
from . import train_b1_late_region_supervision as driver
from .runtime import file_hash

TEXTURE_POOL=driver.m.OUT/'interior-texture-pool.json'


def sample_texture(rng,side,pool):
    row=rng.choice(pool);x0,y0,x1,y1=row['box']
    patch=driver.m.t.r.read(row['path'])[y0:y1,x0:x1]
    labels=driver.m.load_region(row['region'])[y0:y1,x0:x1]
    assert np.unique(labels).size==1 and np.bincount(labels.ravel()).max()==1024
    patch=np.rot90(patch,rng.randrange(4))
    if rng.randrange(2):patch=patch[:,::-1]
    width=rng.randrange(24,49);patch=cv2.resize(patch,(width,width),interpolation=cv2.INTER_LINEAR)
    repeat=side//width+2;tiled=np.tile(patch,(repeat,repeat,1));x=rng.randrange(width);y=rng.randrange(width)
    return np.ascontiguousarray(tiled[y:y+side,x:x+side]),row


def procedural_regions(rng,side,palette,texture):
    """Jittered cells with >=256 native pixels, not a blurred RGB target."""
    grid=4;yy,xx=np.mgrid[:side,:side].astype(np.float32);centres=[]
    for i in range(grid):
        for j in range(grid):
            centres.append(((j+.5+rng.uniform(-.3,.3))*side/grid,
                            (i+.5+rng.uniform(-.3,.3))*side/grid))
    distance=np.stack([(xx-x)**2+(yy-y)**2 for x,y in centres])
    labels=distance.argmin(0).astype(np.int32)
    choices=np.array([rng.randrange(len(palette)) for _ in centres]);rgb=palette[choices[labels]]
    labels,_=area_merge(components(labels),rgb,256)
    labels=boundary_mode(labels,1)
    labels,_=area_merge(components(labels),rgb,256)
    chosen=np.array([rng.randrange(len(palette)) for _ in range(labels.max()+1)])
    codes=chosen[labels];target=palette[codes].astype(np.float32)
    # High-pass real texture tests a different frequency family on every crop.
    texture=texture.astype(np.float32)
    detail=texture-cv2.GaussianBlur(texture,(17,17),3)
    generator=np.random.default_rng(rng.randrange(2**32))
    noise=generator.normal(0,rng.uniform(.25,2.5),target.shape)
    exposure=rng.uniform(-6,6)+rng.uniform(-4,4)*(xx-side/2)/side
    source=target+rng.uniform(.4,1.2)*detail+noise+exposure[...,None]
    return source.round().clip(0,255).astype(np.uint8),codes.astype(np.int64),labels.astype(np.int64)


class TextureRegionTraining(BalancedBoundaryTraining):
    def __init__(self,args,data):
        super().__init__(args,data)
        pool=json.loads(TEXTURE_POOL.read_text(encoding='utf-8'))
        assert pool['train_only'] and pool['data_sha256']==file_hash(driver.m.OUT/'data.json')
        assert {r['id'] for r in pool['patches']}<={r['id'] for r in data['training']}
        self.texture_pool=pool['patches'];self.texture_pool_sha256=file_hash(TEXTURE_POOL)

    def batch_style(self,native):
        # Every fourth update remains a real adjacent video pair. One third
        # teaches known geometry; all other updates keep real teacher examples.
        if not native and self.step%6 in (2,3) and self.step%4!=0:
            sources=[];codes=[];labels=[];specs=[]
            palette=(self.palette.detach().cpu().numpy()*128+128).round().clip(0,255).astype(np.uint8)
            for _ in range(self.args.batch):
                side=self.args.crop;texture,row=sample_texture(self.rng,side,self.texture_pool)
                rgb,code,label=procedural_regions(self.rng,side,palette,texture)
                sources.append(driver.m.t.r.tensor(rgb));codes.append(torch.from_numpy(code).cuda())
                labels.append(torch.from_numpy(label).cuda())
                specs.append(dict(id='synthetic:'+row['id'],xy=row['box'][:2],wh=[side,side],flip=0,
                    texture_patch_box=row['box'],texture_source_sha256=row['sha256'],training_only=True))
            self.class_labels=torch.stack(codes);self.labels=torch.stack(labels)
            return torch.stack(sources),colour_target(self.class_labels,self.palette),specs,None,None
        x,t,specs,flow,valid=super().batch_style(native)
        if not native and flow is None and self.rng.randrange(2):
            # One paired affine colour transform per image leaves label geometry
            # unchanged. It expands missing chroma/luminance coverage without
            # using held-out scenes. The input still goes through real rounding.
            selected=[];transformed=[]
            pal=self.palette*128+128
            for i in range(len(x)):
                gain=self.rng.uniform(.80,1.15);y=self.rng.uniform(-16,12)
                co=self.rng.uniform(-30,15);cg=self.rng.uniform(-16,10)
                offset=x.new_tensor([y+co-cg,y+cg,y-co-cg])
                transformed.append((((x[i]+1)*128*gain+offset[:,None,None]).clamp(0,255)-128)/128)
                changed=(pal*gain+offset).clamp(0,255)
                mapping=(changed[:,None]-pal[None]).square().sum(-1).argmin(-1)
                selected.append(mapping[self.class_labels[i]])
                specs[i]['paired_colour_transform']=dict(gain=gain,y=y,co=co,cg=cg)
            x=torch.stack(transformed);self.class_labels=torch.stack(selected)
            t=colour_target(self.class_labels,self.palette)
        return x,t,specs,flow,valid

    def additional_loss(self,p,target,native):
        loss,values=super().additional_loss(p,target,native)
        # Make all membership decisions coherent inside a labelled colour area.
        # Unlike RGB TV this does not penalize the boundary between two areas.
        logits=(self.nodes['up1_high_dw'][:,:15]-63.5)/4
        labels=self.labels[:,::2,::2]
        if not native:logits=logits[...,20:-20,20:-20];labels=labels[...,20:-20,20:-20]
        probability=logits.sigmoid();flats=[]
        for prob,lab in zip(probability,labels):
            _,idx,count=lab.flatten().unique(return_inverse=True,return_counts=True)
            ids=idx[None].expand(15,-1);mu=prob.new_zeros(15,len(count)).scatter_add(1,ids,prob.flatten(1))/count[None]
            flats.append((prob.flatten(1)-mu[:,idx]).abs().mean())
        flat=torch.stack(flats).mean();values['region_membership_flatness']=flat
        return loss+.5*flat,values

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(texture_to_region_training=True,paired_colour_augmentation=True,
                 procedural_minimum_area_training_only=256,texture_pool_sha256=self.texture_pool_sha256,
                 texture_sampling='32x32 wholly within large train-only target region; tiled with scale and rotation')
        torch.save(s,path)


def regression():
    import random
    palette=np.array([[24,36,60],[200,130,80],[95,145,110]],np.uint8)
    texture=np.random.default_rng(8).integers(0,256,(320,320,3),dtype=np.uint8)
    a=procedural_regions(random.Random(17),320,palette,texture)
    b=procedural_regions(random.Random(17),320,palette,texture)
    assert all(np.array_equal(x,y) for x,y in zip(a,b))
    _,codes,labels=a;assert np.bincount(labels.ravel()).min()>=256
    for label in np.unique(labels):assert np.unique(codes[labels==label]).size==1
    assert np.abs(a[0].astype(float)-palette[codes]).mean()>1
    print('TEXTURE_REGION_REGRESSION_PASS',dict(regions=int(labels.max()+1),
        minimum_area=int(np.bincount(labels.ravel()).min()),training_only=True),flush=True)


@torch.inference_mode()
def preview(net,rows,dest):
    import random
    from PIL import Image,ImageDraw,ImageFont
    from .runtime import write_json
    natural_preview(net,rows,dest)
    pool=json.loads(TEXTURE_POOL.read_text(encoding='utf-8'))['patches']
    pal=actual_palette(net);palette=(pal.cpu().numpy()*128+128).round().clip(0,255).astype(np.uint8)
    board=Image.new('RGB',(960,356*2),'#17202a');draw=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14);records=[]
    for i in range(2):
        texture,row=sample_texture(random.Random(2026100627+i),320,pool)
        rgb,codes,labels=procedural_regions(random.Random(2026100609+i),320,palette,texture)
        x=driver.m.t.r.tensor(rgb)[None];p=net(x)[0][0];target=pal[torch.from_numpy(codes).cuda()].permute(2,0,1)
        for j,(z,title) in enumerate(((x[0],'纹理干扰输入·诊断'),(target,'已知平坦区域·目标'),(p,'真实 B1 输出'))):
            board.paste(driver.m.t.r.c.rgb_image(z),(320*j,356*i+36))
            draw.text((320*j+4,356*i+7),title,font=font,fill='white')
        records.append(dict(seed=2026100609+i,mae=float((p[...,40:-40,40:-40]-target[...,40:-40,40:-40]).abs().mean()*128),
            minimum_label_area=int(np.bincount(labels.ravel()).min())))
    board.save(dest/'synthetic-task-diagnostic.png')
    write_json(dest/'synthetic-task-diagnostic.json',dict(records=records,not_real_scene_acceptance=True,
        texture_pool_sha256=file_hash(TEXTURE_POOL)))


if __name__=='__main__':
    regression();here=Path(__file__).parent
    driver.main(training_factory=TextureRegionTraining,extra_sources=[Path(__file__),
        here/'train_b1_balanced_region_boundaries.py',here/'train_b1_independent_region_boundaries.py',
        here/'train_b1_binary_regions.py',here/'train_b1_ordered_regions.py',here/'minimum_area_blocks.py',
        here/'build_b1_ordered_palette.py',here/'train_b1_quantum_regions.py',here/'train_b1_region_formation.py',
        here/'prepare_b1_texture_patch_pool.py'],
        evaluation=evaluate,preview=preview)
