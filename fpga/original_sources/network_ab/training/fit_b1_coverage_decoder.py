"""Fit only the existing B1 decoder to exact region-footprint coverage.

This is an oracle-latent capacity experiment, never a source-style preview.
The thirty inputs encode actual 4x4 footprint coverage, rather than a hard
180p region decision. Spare decoder channels are active, not silently dead.
"""
import argparse,copy,json,math,random,shutil,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .build_b1_thirty_binary_regions import code_bits
from .probe_b1_ternary_field_targets import palette as original_palette
from .train_b1_ternary_regions import region_codes
from .probe_b1_decoder_coverage import fragment
from .train_b1_joint_region_colour import joint_region_loss
from .train_b1_quantum_regions import RowScaledAdam
from .train_b1_late_region_supervision import executed_sources
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible


def decode(net,z):
    if net.quantized:return fragment(net,z,'res3_add')['rgb_head']/128
    active=False;nodes={'res3_add':z}
    for r in net.plan['nodes']:
        name=r['id'];key=name.replace('.','__')
        if name=='res3_add':active=True;continue
        if not active:continue
        x=nodes[r['inputs'][0]]
        if key in net.weight:
            if r['kernel']==3:x=F.pad(x,(1,1,1,1),mode='replicate')
            y=F.conv2d(x,net.weight[key],net.bias[key],stride=r['stride'],groups=r['groups']).clamp(-128,127)
            if r['activation']=='relu':y=y.clamp_min(0)
        else:
            assert r['kind']=='nearest_x2';y=x.repeat_interleave(2,-2).repeat_interleave(2,-1)
        nodes[name]=y
    return nodes['rgb_head']/128


@torch.no_grad()
def activate_spare(net):
    # Retain the three useful colour coordinates. Additional channels start
    # as differently-thresholded colour combinations, allowing nonlinear
    # reconstruction without changing a single graph node or channel count.
    key='up1_low_pw';base=net.weight[key][:3].clone()
    for c in range(3,16):
        net.weight[key][c]=base[c%3]*1.5
        net.weight[key][c].add_(torch.randn_like(base[0])*.0008)
        net.bias[key][c]=-(c//3)*14
    kernel=net.weight['up1_high_dw'][0].clone()
    net.weight['up1_high_dw'][3:]=kernel
    net.weight['up2_low_dw'].zero_();net.weight['up2_low_dw'][:,0,1,1]=1
    for c in range(3,8):
        net.weight['up2_low_pw'][c].zero_()
        net.weight['up2_low_pw'][c,3+(c-3)*2,0,0]=.65
        net.bias['up2_low_pw'][c]=4
    net.weight['up2_high_dw'][3:]=net.weight['up2_high_dw'][0].clone()
    net.weight['rgb_head'][:,3:].normal_(0,.005)


def objective(p,t):
    colour=F.l1_loss(p,t)
    gradient=sum(F.l1_loss(torch.diff(p,dim=a),torch.diff(t,dim=a)) for a in (-2,-1))/2
    # Aligned footprints supply the true boundary position, unlike source
    # prediction. Exact signed gradients are meaningful for this subproblem.
    return colour+3*gradient,dict(colour=colour,gradient=gradient)


class Fit:
    def __init__(self,args,data):
        self.args=args;self.rows=data['training'];self.rng=random.Random(7317);self.step=0
        self.state=torch.load(args.template,map_location='cpu',weights_only=True)
        ref=torch.load(args.palette_checkpoint,map_location='cpu',weights_only=True)
        pn=m.t.r.c.ColourB1(revive=False);pn.load_state_dict(ref['model']);pn.quantized=True
        pn.package['residual_encoding']=ref['residual_encoding'];self.pk=original_palette(pn)
        self.net=m.t.r.c.ColourB1(revive=False).cuda();self.net.load_state_dict(self.state['model'])
        self.net.package['residual_encoding']=copy.deepcopy(self.state['residual_encoding'])
        activate_spare(self.net);self.net.quantized=False;self.keys=[];self.cache={}
        for key,w in self.net.weight.items():
            active=key.startswith(('up1_','up2_','rgb_head'))
            w.requires_grad_(active);self.net.bias[key].requires_grad_(active)
            if active:self.keys.append(key)
        self.new_optimizer(False)
        self.groups={}
        for row in self.rows:self.groups.setdefault(row['video'],[]).append(row)

    def new_optimizer(self,qat):
        groups=[]
        for key in self.keys:
            w=self.net.weight[key];b=self.net.bias[key]
            if qat:
                s=(getattr(self.net,key+'_m')/131072).float().flatten()
                groups.extend([dict(params=[w],lr=.0004,row_gain=(s*256)[:,None,None,None]),dict(params=[b],lr=.008)])
            else:groups.extend([dict(params=[w],lr=.0004),dict(params=[b],lr=.01)])
        self.opt=RowScaledAdam(groups);self.base_lrs=[g['lr'] for g in groups]

    def sample(self,row):
        if row['id'] not in self.cache:
            codes=region_codes(row['target'],row['region'],self.pk).astype(np.int64)
            target=m.t.r.read(row['target']).copy()
            self.cache[row['id']]=(codes,target)
        return self.cache[row['id']]

    def inputs(self,codes,target):
        ids=torch.from_numpy(np.ascontiguousarray(codes)).cuda()[None]
        z=F.pad((F.avg_pool2d(code_bits(ids),4)*127).round(),(0,0,0,0,0,2))
        t=m.t.r.tensor(np.ascontiguousarray(target))[None]
        return z,t

    def update(self,native=False):
        if self.step==self.args.fp32_steps:
            with torch.no_grad():
                for key in self.keys:
                    w=self.net.weight[key]
                    mm=(w.abs().flatten(1).amax(1).clamp_min(.001)*131072/112).ceil().clamp(1,131071)
                    getattr(self.net,key+'_m').copy_(mm.double()[None,:,None,None])
            self.net.quantized=True;self.new_optimizer(True)
        zs=[];ts=[];samples=[]
        for _ in range(1 if native else 3):
            row=self.rng.choice(self.groups[self.rng.choice(sorted(self.groups))]);codes,target=self.sample(row)
            if not native:
                h,w=codes.shape;xx=4*self.rng.randrange((w-320)//4+1);yy=4*self.rng.randrange((h-320)//4+1)
                codes=codes[yy:yy+320,xx:xx+320];target=target[yy:yy+320,xx:xx+320]
            else:xx=yy=0
            flip=bool(self.rng.randrange(2))
            if flip:codes=codes[:,::-1];target=target[:,::-1]
            z,t=self.inputs(codes,target);zs.append(z);ts.append(t);samples.append([row['id'],xx,yy,flip])
        p=decode(self.net,torch.cat(zs));target=torch.cat(ts)
        if not native:p=p[...,16:-16,16:-16];target=target[...,16:-16,16:-16]
        loss,values=objective(p,target);self.opt.zero_grad(set_to_none=True);loss.backward()
        assert all(w.grad is None or torch.isfinite(w.grad).all() for w in self.net.parameters())
        torch.nn.utils.clip_grad_norm_([p for p in self.net.parameters() if p.requires_grad],10.)
        self.opt.step()
        if self.net.quantized:
            with torch.no_grad():
                for key in self.keys:
                    w=self.net.weight[key];limit=127*(getattr(self.net,key+'_m')/131072).float().flatten()[:,None,None,None]
                    w.copy_(w.maximum(-limit).minimum(limit))
        self.step+=1
        return dict(step=self.step,loss=float(loss.detach()),samples=samples,qat=self.net.quantized,
                    **{k:float(v.detach()) for k,v in values.items()})

    def save(self,path):
        torch.save(dict(model=self.net.state_dict(),optimizer=self.opt.state_dict(),step=self.step,
            residual_encoding=self.net.package['residual_encoding'],rng=self.rng.getstate(),qat=self.net.quantized,
            contract_sha256=self.state['contract_sha256'],thirty_coverage_regions=True,
            template_only=True,source_prediction=False,visual_acceptance=False,args=vars(self.args)),path)

    def restore(self,path):
        state=torch.load(path,map_location='cuda',weights_only=True);self.net.load_state_dict(state['model'])
        self.step=state['step'];self.net.quantized=state['qat'];self.new_optimizer(state['qat'])
        self.opt.load_state_dict(state['optimizer']);self.rng.setstate(state['rng'])


@torch.inference_mode()
def evaluate(fit,rows,dest):
    dest.mkdir(exist_ok=True);images={};records=[]
    for row in rows:
        z,t=fit.inputs(*fit.sample(row));p=decode(fit.net,z)
        labels=torch.from_numpy(m.load_region(row['region']).astype(np.int64)).cuda()[None]
        _,vals=joint_region_loss(p,t,labels);loss,extra=objective(p,t)
        records.append(dict(id=row['id'],loss=float(loss),mae=float((p-t).abs().mean()*128),
                            **{k:float(v) for k,v in vals.items()},**{k:float(v) for k,v in extra.items()}))
        image=m.t.r.c.rgb_image(p[0]);image.save(dest/(row['id']+'_injection.png'));images[row['id']]=image
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14);by_id={r['id']:r for r in rows}
    board=Image.new('RGB',(840,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        for j,(title,img) in enumerate([('target',Image.open(by_id[key]['target'])),('fitted coverage injection',images[key])]):
            board.paste(img.crop(box),(j*420,i*314+34));draw.text((j*420+4,i*314+4),label+' | '+title,font=font,fill='white')
    board.save(dest/'comparison.png');review=dict(step=fit.step,qat=fit.net.quantized,records=records,
        mean={k:sum(r[k] for r in records)/len(records) for k in records[0] if k!='id'},source_prediction=False)
    write_json(dest/'review.json',review);return review


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--template',required=True);ap.add_argument('--palette-checkpoint',required=True)
    ap.add_argument('--data',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--fp32-steps',type=int,default=500);ap.add_argument('--steps',type=int,default=1000)
    ap.add_argument('--interval',type=int,default=250);args=ap.parse_args()
    configure_workspace();contract_guard();set_reproducible(7317);torch.set_num_threads(2)
    dest=Path(args.out);dest.mkdir(exist_ok=False);data=m.load_data(args.data)
    paths,_=executed_sources(Fit);frozen={str(p):file_hash(p) for p in paths}
    for p in paths:shutil.copy2(p,dest/p.name)
    write_json(dest/'config.json',dict(args=vars(args),code=frozen,data_sha256=file_hash(args.data),
        template_sha256=file_hash(args.template),palette_sha256=file_hash(args.palette_checkpoint),source_prediction=False))
    a=Fit(args,data);native=a.update(True);crop=a.update(False);a.save(dest/'preflight.pt')
    expected=a.update(False);expected_state=copy.deepcopy(a.net.state_dict())
    b=Fit(args,data);b.restore(dest/'preflight.pt');actual=b.update(False)
    err=max(float((v-expected_state[k]).abs().max()) for k,v in b.net.state_dict().items())
    assert err==0 and actual['samples']==expected['samples']
    a.step=args.fp32_steps;a.update(False);a.save(dest/'qat-preflight.pt')
    expected=a.update(False);expected_state=copy.deepcopy(a.net.state_dict())
    b.restore(dest/'qat-preflight.pt');actual=b.update(False)
    qat_err=max(float((v-expected_state[k]).abs().max()) for k,v in b.net.state_dict().items())
    assert qat_err==0 and actual['samples']==expected['samples']
    # Integer fragment must equal the existing full graph, node for node.
    a.net.quantized=True
    with torch.no_grad():
        x=torch.randn(1,3,64,64,device='cuda').clamp(-1,127/128)
        full=a.net(x,all_nodes=True);part=fragment(a.net,full['res3_add'],'res3_add')
        errors={k:float((v-full[k]).abs().max()) for k,v in part.items() if k not in ('res1_add','res2_add')}
    assert max(errors.values())==0
    write_json(dest/'preflight.json',dict(native=native,crop=crop,resume_error=err,qat_resume_error=qat_err,integer_fragment_errors=errors,status='PASS'))
    print('COVERAGE_DECODER_PREFLIGHT_PASS',err,flush=True);del a,b;torch.cuda.empty_cache()
    fit=Fit(args,data);evaluate(fit,data['validation'],dest/'initial')
    started=time.perf_counter()
    with (dest/'metrics.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        while fit.step<args.steps:
            row=fit.update(native=fit.step%8==0);log.write(json.dumps(row)+'\n')
            if fit.step%100==0:print('COVERAGE_DECODER_STEP',fit.step,row['loss'],time.perf_counter()-started,flush=True)
            if fit.step%args.interval==0 or fit.step==args.steps:
                assert all(file_hash(p)==h for p,h in frozen.items())
                fit.save(dest/f'candidate_{fit.step:06d}.pt')
                review=evaluate(fit,data['validation'],dest/f'preview_{fit.step:06d}')
                print('COVERAGE_DECODER_REVIEW',fit.step,review['mean'],flush=True)
    print('COVERAGE_DECODER_COMPLETE',flush=True)


if __name__=='__main__':main()
