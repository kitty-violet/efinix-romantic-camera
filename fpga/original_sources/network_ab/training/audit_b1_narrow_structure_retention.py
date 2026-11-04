"""Measure target-selected 1--4px strips/branches on unencoded native720 PNGs.

This complements the straight-step width audit, which explicitly excludes
thin structures. Both sides must have similar target RGB and sufficient
support. This is a bounded profile measurement, not object recognition.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from .runtime import file_hash,write_json
from . import train_b1_minimum_area as m


def selected(labels,hard,soft,maximum=192):
    result=[]
    for axis in (0,1):
        lab=labels if axis==1 else labels.T
        rgb=hard if axis==1 else hard.transpose(1,0,2)
        srgb=soft if axis==1 else soft.transpose(1,0,2)
        parts=[]
        for y in range(6,lab.shape[0]-6,3):
            bounds=np.r_[0,np.flatnonzero(np.diff(lab[y])!=0)+1,lab.shape[1]]
            for a,b in zip(bounds[:-1],bounds[1:]):
                width=b-a
                if not 1<=width<=4 or a<12 or b>lab.shape[1]-12:continue
                if not (lab[y,a-4:a]==lab[y,a-1]).all() or not (lab[y,b:b+4]==lab[y,b]).all():continue
                left=rgb[y,a-4:a].mean(0);right=rgb[y,b:b+4].mean(0)
                if abs(left-right).max()>10:continue
                colour=rgb[y,a:b].mean(0)-(left+right)/2
                if abs(colour).max()<16:continue
                reference=srgb[y,a-8:b+8].astype(float)
                base=np.r_[reference[:4],reference[-4:]].mean(0)
                contrast=reference[8:8+width].mean(0)-base
                if np.linalg.norm(contrast)<12:continue
                if parts and abs(y-parts[-1][0])<5 and abs(a-parts[-1][1])<5:continue
                parts.append((y,int(a),int(width)))
        if len(parts)>maximum:parts=[parts[k] for k in np.linspace(0,len(parts)-1,maximum).astype(int)]
        result.extend((axis,*v) for v in parts)
    return result


def measure(image,reference,y,a,width):
    ref=reference[y,a-8:a+width+8].astype(float);dc=ref.mean(0)
    trials=[image[y,a-8+shift:a+width+8+shift].astype(float) for shift in range(-2,3)]
    scores=[np.abs((q-q.mean(0))-(ref-dc)).mean() for q in trials]
    chosen=int(np.argmin(scores));p=trials[chosen]
    base=np.r_[ref[:4],ref[-4:]].mean(0);v=ref[8:8+width].mean(0)-base;vv=float(v@v)
    pb=np.r_[p[:4],p[-4:]].mean(0);pv=p[8:8+width].mean(0)-pb
    ratio=float(pv@v/vv)
    pp=(p-pb)@v/vv;tt=(ref-base)@v/vv
    outside=np.ones(len(p),bool);outside[7:9+width]=False
    return dict(relative_contrast=ratio,aligned_profile_mae=float(scores[chosen]),offset=chosen-2,
        outside_ridge_error=float(np.abs(pp[outside]-tt[outside]).mean()),
        chromatic_direction_error=float(np.linalg.norm(pv-ratio*v)))


def regression():
    for width in (1,2,3,4):
        x=np.full((48,96,3),60.,float);x[:,48:48+width]=[100,120,140]
        a=measure(x,x,24,48,width);assert a['relative_contrast']==1 and a['aligned_profile_mae']==0
        zero=np.full_like(x,60);assert measure(zero,x,24,48,width)['relative_contrast']==0
        roll=np.roll(x,2,axis=1);assert measure(roll,x,24,48,width)['aligned_profile_mae']==0
        blur=np.stack([np.apply_along_axis(lambda z:np.convolve(np.pad(z,(4,4),mode='edge'),np.ones(9)/9,mode='valid'),1,x[...,c]) for c in range(3)],-1)
        assert measure(blur,x,24,48,width)['relative_contrast']<.6
    print('NARROW_STRUCTURE_PASS exact/absent/shift2/blur widths1to4',flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--previews',nargs='+',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    regression();d=m.load_data(a.data);allrows=[];hashes={}
    for row in d['validation']:
        target=m.t.r.read(row['target']);hard=m.t.r.read(row['hard_region_target']);labels=m.load_region(row['region'])
        choices={'target':target}
        for folder in a.previews:
            path=Path(folder)/(row['id']+'_output.png');name=Path(folder).parent.name+'/'+Path(folder).name
            choices[name]=m.t.r.read(path);hashes[str(path)]=file_hash(path)
        records=[]
        for axis,y,x,width in selected(labels,hard,target):
            orient=lambda z:z if axis==1 else z.transpose(1,0,2)
            records.append(dict(axis=axis,xy=[x,y],width=width,values={name:measure(orient(im),orient(target),y,x,width) for name,im in choices.items()}))
        allrows.append(dict(id=row['id'],records=records));print('NARROW_SELECTED',row['id'],len(records),flush=True)
    summary={}
    for name in choices:
        rr=[r['values'][name] for row in allrows for r in row['records']];assert rr
        contrast=np.array([r['relative_contrast'] for r in rr])
        summary[name]=dict(samples=len(rr),median_contrast=float(np.median(contrast)),fraction_below_quarter=float(np.mean(contrast<.25)),
            fraction_below_half=float(np.mean(contrast<.5)),outside_ridge_error=float(np.mean([r['outside_ridge_error'] for r in rr])),
            chromatic_direction_error=float(np.mean([r['chromatic_direction_error'] for r in rr])))
    out=Path(a.out);out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name)
    write_json(out/'results.json',dict(summary=summary,records=allrows,png_sha256=hashes,
        scope='Target-selected one-to-four-pixel strips with similar RGB flanks; shared RGB displacement at most2; no semantic or video acceptance.'))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
