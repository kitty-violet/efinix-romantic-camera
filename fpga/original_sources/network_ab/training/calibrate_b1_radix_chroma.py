"""TRAIN-only colour calibration with unchanged B1 contour weights.

The five already allocated spare channels carry hinges of the existing Cg
coordinate. This allows near-neutral chroma correction without uniformly
desaturating foliage. Only up2_low_pw spare rows, up2_high_dw spare rows and
rgb_head change; every variant is replayed from the actual20KiB image.
No inference LUT, semantic masks, filtering pass or additional B1 operation.
"""
import argparse,copy
from pathlib import Path
import cv2,numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .build_b1_thirty_binary_regions import code_bits
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


@torch.no_grad()
def add_hinges(net):
    for j,knot in enumerate((2,3,4,5,6),3):
        net.weight['up2_low_pw'][j]=net.weight['up2_low_pw'][2]
        net.bias['up2_low_pw'][j]=net.bias['up2_low_pw'][2]-8*knot
        net.up2_low_pw_m[:,j]=net.up2_low_pw_m[:,2]
        net.weight['up2_high_dw'][j]=net.weight['up2_high_dw'][2]
        net.bias['up2_high_dw'][j]=net.bias['up2_high_dw'][2]
        net.up2_high_dw_m[:,j]=net.up2_high_dw_m[:,2]


@torch.inference_mode()
def constants(net):
    z=F.pad(code_bits(torch.arange(1215)[:,None,None])*127,(0,0,0,0,0,2))
    nodes=fragment(net,z,'up1_low_dw')
    return nodes['up2_high_dw'][:,:,0,0].double(),nodes['rgb_head'][:,:,0,0].double()


@torch.no_grad()
def fit_head(net,solution):
    w=solution[:8].T.float();bias=solution[8].float()
    net.weight['rgb_head'][:,:,0,0]=w;net.bias['rgb_head'].copy_(bias)
    mult=(w.abs().amax(1)*131072/120).ceil().clamp(1,131071)
    net.rgb_head_m.copy_(mult.double()[None,:,None,None])


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(2)
    out=Path(a.out);out.mkdir(exist_ok=False);state,base=load(a.checkpoint);data=m.load_data(a.data)
    enhanced=copy.deepcopy(base);add_hinges(enhanced)
    feature,original_palette=constants(enhanced);_,expected=constants(base)
    assert torch.equal(original_palette,expected)
    design=torch.cat((feature,torch.ones(1215,1,dtype=torch.float64)),1)
    targets=[];indices=[];weights=[];counts={}
    for row in data['training']:counts[row['video']]=counts.get(row['video'],0)+1
    for i,row in enumerate(data['training']):
        rgb=m.t.r.read(row['target']).astype(np.float64);lab=m.load_region(row['region']).astype(np.int64)
        size=np.bincount(lab.ravel());keep=size>0
        mean=np.stack([np.bincount(lab.ravel(),weights=rgb[...,c].ravel(),minlength=len(size))/size.clip(1) for c in range(3)],1)[keep]-128
        target=torch.from_numpy(mean);ids=[]
        for part in target.split(2048):
            d=part.square().sum(1)[:,None]+expected.square().sum(1)[None]-2*part@expected.T
            ids.append(d.argmin(1))
        weight=np.sqrt(size[keep]);weight/=weight.sum()*counts[row['video']]
        targets.append(target);indices.append(torch.cat(ids));weights.append(torch.from_numpy(weight))
        if (i+1)%50==0:print('RADIX_CHROMA_DATA',i+1,flush=True)
    target=torch.cat(targets);ids=torch.cat(indices);weight=torch.cat(weights);weight/=weight.sum()
    x=design[ids];gram=x.T@(x*weight[:,None]);cross=x.T@(target*weight[:,None])
    prior=torch.cat((enhanced.weight['rgb_head'][:,:,0,0].T,enhanced.bias['rgb_head'][None]),0).double()
    # Penalize colour changes on the full existing palette, so rarely used
    # code combinations cannot acquire unconstrained colour extrapolations.
    anchor=design.T@design/len(design)
    variants={'before':base};fit=[]
    for mode,lam in (('linear',.01),('hinges',.01),('hinges_guarded',.05),('hinges_iterative',.01)):
        cols=torch.tensor((0,1,2,8)) if mode=='linear' else torch.arange(9)
        g=gram[cols][:,cols]+lam*anchor[cols][:,cols]
        rhs=cross[cols]+lam*(anchor@prior)[cols]
        solution=prior.clone();solution[cols]=torch.linalg.solve(g+torch.eye(len(cols),dtype=torch.float64)*1e-8,rhs)
        # All solutions are scored after real INT8/Q17 quantization.
        net=copy.deepcopy(enhanced);fit_head(net,solution);_,pal=constants(net);fit_history=[]
        if mode=='hinges_iterative':
            for iteration in range(12):
                assigned=[]
                for part in target.split(2048):
                    d=part.square().sum(1)[:,None]+pal.square().sum(1)[None]-2*part@pal.T
                    assigned.append(d.argmin(1))
                closest=torch.cat(assigned);fit_history.append(dict(iteration=iteration,mae=float(((pal[closest]-target).abs().mean(1)*weight).sum())))
                xx=design[closest];gg=xx.T@(xx*weight[:,None]);cc=xx.T@(target*weight[:,None])
                solution=torch.linalg.solve(gg+lam*anchor+torch.eye(9,dtype=torch.float64)*1e-8,cc+lam*anchor@prior)
                fit_head(net,solution);_,pal=constants(net)
        delta=pal[ids]-target;err=float((delta.abs().mean(1)*weight).sum())
        lev=pal.reshape(15,9,9,3);cg=(2*lev[...,1]-lev[...,0]-lev[...,2])/4
        changes=[k for k,v in net.state_dict().items() if not torch.equal(v,base.state_dict()[k])]
        assert all(any(n in k for n in ('up2_low_pw','up2_high_dw','rgb_head')) for k in changes)
        assert torch.equal(net.weight['up2_low_pw'][:3],base.weight['up2_low_pw'][:3])
        fit.append(dict(mode=mode,ridge=lam,train_mae_fixed_original_codes=err,iterative_history=fit_history,coefficient=solution.tolist(),
            actual_cg_curve_at_y7_co4=cg[7,4].tolist(),changed_keys=changes))
        variants[mode]=net
    fit.insert(0,dict(mode='before',train_mae=float(((expected[ids]-target).abs().mean(1)*weight).sum())))
    shots={};records=[]
    for name,net in variants.items():
        raw=net.export();assert len(raw)==20480;actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu')
        sub=out/name;sub.mkdir();ss={k:v for k,v in state.items() if k not in ('optimizer','rng','torch_rng','cuda_rng')}
        ss.update(model=copy.deepcopy(net.state_dict()),radix_colour_calibration=name,
            calibration_training_ids=[r['id'] for r in data['training']],constant_palette_requires_runtime=True,
            five_existing_cg_hinges=True,extra_inference_operations=0,visual_acceptance=False)
        torch.save(ss,sub/'candidate.pt');(sub/'coefficients-padded.bin').write_bytes(raw)
        for row in data['validation']:
            inp=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float()/128-1
            nn=net(inp,all_nodes=True);other,_=actual(inp*128,True)
            assert all(torch.equal(v,other[k]) for k,v in nn.items())
            p=other['rgb_head'][0].permute(1,2,0).numpy()+128;t=m.t.r.read(row['target']).astype(np.float32)
            for rid,box,title in m.t.r.c.CROPS:
                if rid!=row['id']:continue
                xx,yy,xe,ye=box;d=(p-t)[yy:ye,xx:xe]
                records.append(dict(mode=name,id=rid,crop=title,mae=float(np.abs(d).mean()),Cg_bias=float(((2*d[...,1]-d[...,0]-d[...,2])/4).mean())))
                shots[name,rid,title]=Image.fromarray(p.astype(np.uint8)).crop(box)
            print('RADIX_CHROMA_REPLAY',name,row['id'],flush=True)
    board=Image.new('RGB',(420*(len(variants)+1),1256),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12);by={r['id']:r for r in data['validation']}
    for i,(rid,box,title) in enumerate(m.t.r.c.CROPS):
        pics=[Image.open(by[rid]['target']).crop(box)]+[shots[name,rid,title] for name in variants]
        for j,(im,label) in enumerate(zip(pics,('目标','当前整数B1','线性校色','中性色分段校色','较强全色表约束','迭代色表·未重新训练'))):
            board.paste(im,(j*420,i*314+34));draw.text((j*420+3,i*314+4),title+' | '+label,font=font,fill='white')
    board.save(out/'comparison.png')
    write_json(out/'review.json',dict(fit=fit,records=records,source_sha256=file_hash(__file__),checkpoint_sha256=file_hash(a.checkpoint),data_sha256=file_hash(a.data),
        training_regions=len(target),source_videos=len(counts),all_nodes_integer_error=0,bytes=20480,train_only=True,external_filter=False,visual_acceptance=False))
    print('RADIX_CHROMA_COMPLETE',fit,flush=True)


if __name__=='__main__':main()
