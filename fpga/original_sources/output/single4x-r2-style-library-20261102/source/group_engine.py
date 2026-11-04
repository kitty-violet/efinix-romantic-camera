"""Batch independent exported networks via channel groups; same integer pixels."""
import copy
import os
import library as l
from library import torch,F,HalfAway,PackedEngine,IntegerEngine

class GroupEngine:
    def __init__(self,folders=None,entries=None):
        assert (folders is None)!=(entries is None)
        if folders is not None:
            engines=[PackedEngine(p) for p in folders];self.rows=engines[0].rows
            residuals=[e.residual for e in engines];self.output='rgb'
            assert all(e.rows==self.rows for e in engines)
        else:
            engines=[IntegerEngine(__import__('pathlib').Path(e['source']).read_bytes()) for e in entries]
            self.rows=[]
            for raw in engines[0].plan['nodes']:
                r=copy.deepcopy(raw);r['cin']=r['input_channels']
                if r['kind']=='nearest_x2':r['kind']='nearest';r['factor']=2
                elif r['kind']=='residual_add':r['kind']='add'
                self.rows.append(r)
            residuals=[e.package['residual_encoding'] for e in engines];self.output='rgb_head'
        self.count=len(engines);self.ops={};self.residual={}
        for n in engines[0].ops:
            self.ops[n]=tuple(torch.cat([e.ops[n][j] for e in engines],dim=0 if j==0 else 1) for j in range(3))
        self.int_gemm=os.environ.get('S4R2_INT_GEMM','1')=='1'
        self.int_weights={}
        if self.int_gemm:
            for r in self.rows:
                n=r['id']
                if n not in self.ops or (n!='stem' and r['kernel']!=1):continue
                w=self.ops[n][0];cin=r['cin']*r['kernel']**2;cout=r['channels'];padded_cin=(cin+7)//8*8
                if n=='stem':
                    b=w.reshape(self.count*cout,cin).T.contiguous().to(torch.int8)
                    self.int_weights[n]=F.pad(b,(0,0,0,padded_cin-cin)).contiguous()
                else:
                    bs=[]
                    for i in range(self.count):
                        b=w[i*cout:(i+1)*cout].reshape(cout,cin).T.contiguous().to(torch.int8)
                        bs.append(F.pad(b,(0,(cout+7)//8*8-cout,0,padded_cin-cin)).contiguous())
                    self.int_weights[n]=bs
        for r in self.rows:
            if r['kind']!='add':continue
            ps=[p[r['id']] for p in residuals]
            assert len({p['main_scaled'] for p in ps})==1
            aligned=torch.tensor([p['multiplier'] for p in ps],device='cuda',dtype=torch.float64).repeat_interleave(r['channels'])[None,:,None,None]
            post=torch.tensor([p['post_multiplier'] for p in ps],device='cuda',dtype=torch.float64).repeat_interleave(r['channels'])[None,:,None,None]
            self.residual[r['id']]=(ps[0]['main_scaled'],aligned,post)
    @torch.inference_mode()
    def __call__(self,x):
        assert x.shape[0]==1
        v={'input':x*128};uses={}
        for r in self.rows:
            for k in r['inputs']:uses[k]=uses.get(k,0)+1
        for r in self.rows:
            n=r['id'];a=v[r['inputs'][0]]
            if n in self.ops:
                w,b,m=self.ops[n]
                if r['kernel']==3:a=F.pad(a,(1,1,1,1),mode='replicate')
                if n=='stem':
                    # Every style sees the same RGB. Concatenated output
                    # filters avoid redundant grouped RGB convolutions.
                    if self.int_gemm:
                        h,ww=x.shape[-2:];patches=F.unfold(a,(3,3)).squeeze(0).T.contiguous().to(torch.int8)
                        patches=F.pad(patches,(0,32-27,0,(-len(patches))%16)).contiguous()
                        dot=torch._int_mm(patches,self.int_weights[n])[:h*ww].T.reshape(1,self.count*r['channels'],h,ww).float()
                    else:dot=F.conv2d(a,w,stride=r['stride'],groups=1)
                elif r['kernel']==1:
                    h,ww=a.shape[-2:];cin=r['cin'];cout=r['channels']
                    if self.int_gemm:
                        inputs=a.reshape(self.count,cin,h*ww).transpose(1,2).contiguous().to(torch.int8);parts=[]
                        for i in range(self.count):
                            mat=F.pad(inputs[i],(0,(-cin)%8,0,(-h*ww)%16)).contiguous()
                            parts.append(torch._int_mm(mat,self.int_weights[n][i])[:h*ww,:cout].T)
                        dot=torch.stack(parts).reshape(1,self.count*cout,h,ww).float()
                    else:dot=torch.bmm(w.reshape(self.count,cout,cin),a.reshape(self.count,cin,h*ww)).reshape(1,self.count*cout,h,ww)
                else:dot=F.conv2d(a,w,stride=r['stride'],groups=r['groups']*self.count)
                y=HalfAway.apply((dot.double()+b).clamp(-(2**31),2**31-1)*m/131072).clamp(-128,127).float()
                if r['activation']=='relu':y=y.clamp_min(0)
            elif r['kind']=='nearest':y=a.repeat_interleave(r['factor'],-2).repeat_interleave(r['factor'],-1)
            else:
                z=v[r['inputs'][1]];main,align,post=self.residual[n];small,other=(a,z) if main else (z,a)
                aligned=HalfAway.apply(small.double()*align/131072).clamp(-128,127)
                y=HalfAway.apply((aligned+other.double())*post/131072).clamp(-128,127).float()
            v[n]=y
            for k in r['inputs']:
                uses[k]-=1
                if not uses[k]:del v[k]
        return v[self.output].reshape(self.count,3,x.shape[-2],x.shape[-1])/128

if __name__=='__main__':
    from PIL import Image
    import numpy as np
    l.setup(123);entries=l.read_inventory()[:4];a=np.array(Image.open(l.OLD/'rule-preview/outdoor_source.png').convert('RGB'));x=l.tx(a[None])
    count=0
    for mode in ['new','old']:
        engines=[l.PackedEngine(l.D/'initial'/e['id']) for e in entries] if mode=='new' else [l.Teacher(e) for e in entries]
        group=GroupEngine(folders=[l.D/'initial'/e['id'] for e in entries]) if mode=='new' else GroupEngine(entries=entries)
        y=group(x)
        for i,engine in enumerate(engines):assert torch.equal(y[i:i+1],engine(x));count+=y[i].numel()
    l.save_json(l.D/'group-engine-check.json',dict(status='PASS',scalar_comparisons=count,full720=True,group_count=4,no_cross_style_channel_mixing=True))
    print('GROUP_ENGINE_FULL720_EXACT',count,flush=True)
