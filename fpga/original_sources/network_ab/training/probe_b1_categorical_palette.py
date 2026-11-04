"""Training-only capacity check for mutually exclusive B1 colour regions.

Eight final channels can encode eight colour directions with discrete levels.
These figures quantize known targets; they are not source-network predictions.
"""
import argparse
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .runtime import configure_workspace,write_json,file_hash


def nearest(rgb,palette):
    flat=np.asarray(rgb,np.float64).reshape(-1,3);result=[];squared=(palette*palette).sum(1)[None]
    for start in range(0,len(flat),8192):
        x=flat[start:start+8192];distance=(x*x).sum(1)[:,None]+squared-2*x@palette.T
        result.append(distance.argmin(1))
    return np.concatenate(result)


def fit(points,levels,seed,exponent=1.):
    cv2.setRNGSeed(seed)
    _,_,centers=cv2.kmeans(points.astype(np.float32),8,None,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER,60,.1),1,cv2.KMEANS_PP_CENTERS)
    features=np.zeros((1+8*levels,9),np.float64);features[:,-1]=1
    for ray in range(8):features[1+ray*levels:1+(ray+1)*levels,ray]=np.arange(1,levels+1)/levels
    base=np.quantile(points,.025,axis=0).clip(0,64);weights=np.vstack(((centers-base)*1.35,base))
    code=(points.astype(np.int64)//4);packed=code[:,0]*4096+code[:,1]*64+code[:,2]
    unique,counts=np.unique(packed,return_counts=True)
    frequencies=counts.astype(np.float64)**exponent
    colours=np.stack((unique//4096,(unique//64)%64,unique%64),1)*4+2
    # Exponent1 preserves pixel frequencies;0.5 gives rarer training colours
    # more influence. No validation pixels initialize, fit or select palettes.
    best=None
    for _ in range(50):
        palette=(features@weights).clip(0,255);ids=nearest(colours,palette)
        size=np.bincount(ids,weights=frequencies,minlength=len(features))
        sums=np.stack([np.bincount(ids,weights=frequencies*colours[:,c],minlength=len(features)) for c in range(3)],1)
        xx=features.T@(features*size[:,None]);xy=features.T@sums
        ridge=np.eye(9)*frequencies.sum()*1e-5;ridge[-1,-1]*=.01
        loss=float(((colours-palette[ids])**2*frequencies[:,None]).sum()/(3*frequencies.sum()))
        if best is None or loss<best[0]:best=(loss,palette.copy(),weights.copy())
        weights=np.linalg.solve(xx+ridge,xy+ridge@weights)
    # Recompute the actual palette corresponding to the saved coefficients.
    palette=(features@best[2]).clip(0,255);ids=nearest(points,palette)
    loss=float(np.mean((points-palette[ids])**2))
    return dict(levels=levels,seed=seed,train_mse=loss,train_objective=best[0],exponent=exponent,palette=palette,weights=best[2],features=features)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--exponent',type=float,default=1.);a=ap.parse_args()
    configure_workspace();cv2.setNumThreads(2);dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    data=m.load_data();training=data['training'][::4]
    points=np.concatenate([m.t.r.read(r['target'])[::16,::16].reshape(-1,3) for r in training]).astype(np.float64)
    candidates={};records={}
    for levels in (2,4):
        trials=[fit(points,levels,seed,a.exponent) for seed in (271,314,419)]
        best=min(trials,key=lambda r:r['train_objective']);candidates[levels]=best;record=[]
        for row in data['validation']:
            target=m.t.r.read(row['target']);ids=nearest(target,best['palette'])
            rgb=best['palette'][ids].reshape(target.shape).round().clip(0,255).astype(np.uint8)
            Image.fromarray(rgb).save(dest/(row['id']+f'_palette{1+8*levels}_target_only.png'))
            record.append(dict(id=row['id'],mae=float(np.abs(rgb.astype(np.float64)-target).mean())))
        records[levels]=record
        print('CATEGORICAL_PALETTE_CAPACITY',1+8*levels,best['train_mse'],record,flush=True)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14);board=Image.new('RGB',(1260,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    rows={r['id']:r for r in data['validation']}
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        paths=[rows[key]['target']]+[dest/(key+f'_palette{1+8*levels}_target_only.png') for levels in (2,4)]
        for j,(path,title) in enumerate(zip(paths,('离线目标','17色容量诊断·非推理','33色容量诊断·非推理'))):
            board.paste(Image.open(path).crop(box),(j*420,i*314+34));draw.text((j*420+4,i*314+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'target-capacity.png')
    shutil.copy2(__file__,dest/Path(__file__).name)
    serial={str(k):{key:(value.tolist() if isinstance(value,np.ndarray) else value) for key,value in v.items()} for k,v in candidates.items()}
    write_json(dest/'results.json',dict(candidates=serial,records=records,training_ids=[r['id'] for r in training],
        data_sha256=file_hash(m.OUT/'data.json'),target_injection_only=True,network_trained=False,
        hardware_inference_verified=False,visual_acceptance=False))


if __name__=='__main__':main()
