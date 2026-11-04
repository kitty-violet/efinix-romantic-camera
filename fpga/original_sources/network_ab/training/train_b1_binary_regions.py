"""Learn four binary region decisions at360p using exact B1 forward values."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from torch.nn import functional as F
from . import train_b1_minimum_area as m
from . import train_b1_late_region_supervision as driver
from . import train_b1_region_formation as formation
from .train_b1_quantum_regions import RowScaledAdam
from .train_b1_clean_regions import CleanRegionTraining
from .train_b1_winner_surrogate import ExactValueProxyGradient,WinnerSurrogateTraining
from .fit_b1_sixteen_regions import region_class_map,palette_key
from .build_b1_binary_palette import actual_palette,bits
from .train_b1_sixteen_regions import colour_target


class BinaryGradientB1(m.t.r.c.ColourB1):
    def training_node_proxy(self,name,value,nodes):
        if name!='up2_low_dw':return value
        logits=(nodes['up1_high_dw'][:,:4]-63.5)/4
        proxy=F.pad(127*logits.sigmoid(),(0,0,0,0,0,12))
        return ExactValueProxyGradient.apply(value,proxy)


class BinaryTraining(m.AreaTraining):
    checkpoint_flag='binary_regions'
    network_class=BinaryGradientB1
    palette_function=staticmethod(actual_palette)
    bit_function=staticmethod(bits)
    num_bits=4

    def __init__(self,args,data):
        super().__init__(args,data);warm=torch.load(args.warm,map_location='cpu',weights_only=True)
        assert warm.get(self.checkpoint_flag);self.net.package['residual_encoding']=warm['residual_encoding']
        # The reconstruction base recalibrates an FP32 warm start for QAT.
        # This fixed integer decoder already has an audited palette/threshold
        # program: preserve ALL its saved row scales across FP32 -> QAT.
        for key in self.net.weight:
            getattr(self.net,key+'_m').copy_(warm['model'][key+'_m'].to(getattr(self.net,key+'_m').device))
        self.net.__class__=self.network_class;self.palette=self.palette_function(self.net);self.pk=palette_key(self.palette);self.bit_table=self.bit_function('cuda')
        self.fixed={};groups=[];fixed=False
        for key,w in self.net.weight.items():
            if key=='up1_high_dw':fixed=True
            train=not fixed
            w.requires_grad_(train);self.net.bias[key].requires_grad_(train)
            if train:
                s=(getattr(self.net,key+'_m')/131072).float().flatten()
                groups.append(dict(params=[w],lr=args.lr,row_gain=(s/(1/256))[:,None,None,None]))
                groups.append(dict(params=[self.net.bias[key]],lr=.004))
            else:self.fixed[key]=(w.detach().clone(),self.net.bias[key].detach().clone())
        self.opt=RowScaledAdam(groups);self.base_lrs=[g['lr'] for g in groups]
        def remember(module,inputs,result):
            if isinstance(result,dict):self.nodes=result
        self.net.register_forward_hook(remember)

    def batch_style(self,native):
        x,old,specs,flow,valid=super().batch_style(native);codes=[]
        for spec in specs:
            row=self.by_id.get(spec['id']) or self.by_stem[spec['id']]
            a=region_class_map(row['target'],row['region'],self.pk);xx,yy=spec['xy'];w,h=spec['wh'];a=a[yy:yy+h,xx:xx+w]
            if spec['flip']:a=a[:,::-1]
            codes.append(torch.from_numpy(np.ascontiguousarray(a.astype(np.int64))).cuda())
        self.class_labels=torch.stack(codes)
        return x,colour_target(self.class_labels,self.palette),specs,flow,valid
    classification_weights=WinnerSurrogateTraining.classification_weights
    input_stability_loss=CleanRegionTraining.input_stability_loss

    def predicted_codes(self,logits):
        return ((logits>=0).long()*torch.tensor([1,2,4,8],device=logits.device)[:,None,None]).sum(0)

    def additional_loss(self,p,target,native):
        labels=self.labels if native else self.labels[...,40:-40,40:-40]
        codes=self.class_labels if native else self.class_labels[...,40:-40,40:-40]
        shape,values=formation.formation_loss(p,target,labels)
        logits=(self.nodes['up1_high_dw'][:,:self.num_bits]-63.5)/4
        if not native:logits=logits[...,20:-20,20:-20]
        truth=F.avg_pool2d(self.bit_table[codes].permute(0,3,1,2),2)
        weight=self.classification_weights(labels,logits.shape[-2:])
        ce=F.binary_cross_entropy_with_logits(logits,truth,reduction='none').mean(1,keepdim=True)
        cls=(ce*weight).sum()/weight.sum()
        pure=((truth<.01)|(truth>.99)).float();sign=2*truth-1
        margin=(F.relu(1.5-sign*logits)*pure*weight).sum()/(pure*weight).sum().clamp_min(1)
        loss=self.args.area_weight*shape+self.args.classification*(cls+.1*margin)
        values.update(binary_classification=cls,binary_margin=margin)
        if not native and self.step%8==0:
            from .minimum_area_blocks import components,area_merge,boundary_mode
            with torch.no_grad():
                ids=self.predicted_codes(logits[0]).cpu().numpy()
                pal=(self.palette.cpu().numpy()*128+128).round().clip(0,255).astype(np.uint8)
                rgb=pal[ids];merged,_=area_merge(components(ids),rgb,64)
                clean,_=area_merge(components(boundary_mode(merged,1)),rgb,64)
                count=np.bincount(clean.ravel());means=np.stack([np.bincount(clean.ravel(),weights=rgb[...,c].ravel())/count for c in range(3)],1)
                colour=((means[:,None]-pal[None].astype(float))**2).sum(2).argmin(1)[clean]
                correction=torch.from_numpy(colour.astype(np.int64)).cuda()[None]
                changed=(correction!=torch.from_numpy(ids).cuda()[None]).float()[:,None]
                wanted=self.bit_table[correction].permute(0,3,1,2)
            cleanup=(F.binary_cross_entropy_with_logits(logits[:1],wanted,reduction='none')*changed).sum()/(self.num_bits*changed.sum()).clamp_min(1)
            loss+=.3*cleanup;values.update(colour_area_cleanup=cleanup,cleanup_fraction=changed.mean())
        return loss,values

    def update(self,native=None):
        row=super().update(native)
        if self.step%100==0:
            for k,(w,b) in self.fixed.items():assert torch.equal(w,self.net.weight[k]) and torch.equal(b,self.net.bias[k]),k
        return row

    def save(self,path):
        super().save(path);s=torch.load(path,map_location='cpu',weights_only=True)
        s.update(residual_encoding=self.net.package['residual_encoding'],coverage_layout='4binary360_rgb16',binary_regions=True,preserved_warm_quantization_scales=True,
            binary_backward_proxy=True,attainable_region_colour_targets=True,palette=self.palette.cpu().tolist())
        torch.save(s,path)


@torch.inference_mode()
def evaluate_palette(net,rows,palette_function):
    palette=palette_function(net);pk=palette_key(palette);records=[]
    for row in rows:
        x=m.t.r.tensor(m.t.r.read(row['path']))[None];p=net(x)[0]
        codes=torch.from_numpy(region_class_map(row['target'],row['region'],pk).astype(np.int64)).cuda()[None]
        target=colour_target(codes,palette);labels=torch.from_numpy(m.load_region(row['region']).astype(np.int64)).cuda()[None]
        shape,values=formation.formation_loss(p,target,labels);appear,_=m.t.appearance(p,target,'direct')
        metrics=m.t.r.metrics(p,target);metrics.update({k:float(v) for k,v in values.items()})
        original=m.t.r.tensor(m.t.r.read(row['target']))[None]
        metrics.update(score=float(shape+.25*appear),original_target_mae=float((p-original).abs().mean()*128))
        records.append(dict(id=row['id'],**metrics))
    return dict(records=records,mean={k:sum(r[k] for r in records)/len(records) for k in records[0] if k!='id'})


def evaluate(net,rows):return evaluate_palette(net,rows,actual_palette)


@torch.inference_mode()
def preview_palette(net,rows,dest,palette_function):
    dest.mkdir(exist_ok=True);palette=palette_function(net);pk=palette_key(palette);images={}
    for row in rows:
        source=m.t.r.read(row['path']);x=m.t.r.tensor(source)[None];output=net(x)[0][0]
        codes=torch.from_numpy(region_class_map(row['target'],row['region'],pk).astype(np.int64)).cuda()[None]
        target=colour_target(codes,palette)[0]
        pics=[Image.fromarray(source),Image.fromarray(m.t.r.read(row['target'])),m.t.r.c.rgb_image(target),m.t.r.c.rgb_image(output)]
        images[row['id']]=pics
        for label,im in zip(('source','original_target','actual_training_target','actual_B1'),pics):im.save(dest/(row['id']+'_'+label+'.png'))
    board=Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        output_title='真实整数B1输出' if net.quantized else '浮点训练输出·非部署结果'
        for j,(pic,title) in enumerate(zip(images[key],('原图','原区域目标·量化前','本轮实际训练目标',output_title))):
            board.paste(pic.crop(box),(j*420,i*314+34));draw.text((j*420+4,i*314+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'comparison.png')


if __name__=='__main__':
    here=Path(__file__).parent
    driver.main(training_factory=BinaryTraining,extra_sources=[Path(__file__),here/'build_b1_binary_palette.py',
        here/'fit_b1_sixteen_regions.py',here/'train_b1_sixteen_regions.py',here/'train_b1_winner_surrogate.py',
        here/'train_b1_quantum_regions.py',here/'train_b1_clean_regions.py',here/'minimum_area_blocks.py',Path(formation.__file__)],evaluation=evaluate)
