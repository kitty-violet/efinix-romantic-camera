"""Measure RGB transition widths on identical target-selected straight edges.

This diagnostic reads unencoded PNGs. It excludes thin structures/junctions;
the full-profile and natural crop checks remain necessary for those cases.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m


def crossing(q,level):
    ids=np.flatnonzero((q[:-1]<=level)&(q[1:]>level))
    if not len(ids):return None
    at=ids+(level-q[ids])/(q[ids+1]-q[ids])
    return float(at[np.argmin(abs(at-(len(q)-1)/2))])


def measure(profile,reference):
    a=reference[:4].mean(0);b=reference[-4:].mean(0);direction=b-a;length=float(direction@direction)
    assert length>1e-8
    q=(profile-a)@direction/length
    lo=float(q[:4].mean());hi=float(q[-4:].mean());span=hi-lo
    width=None
    if span>.25:
        v=(q-lo)/span;c10=crossing(v,.1);c90=crossing(v,.9)
        if c10 is not None and c90 is not None and c90>=c10:width=c90-c10
    segment=a+q.clip(0,1)[:,None]*direction
    return dict(width10_90=width,relative_contrast=span,
        rgb_off_reference_segment=float(np.sqrt(((profile-segment)**2).mean())),
        backtracking=float(np.maximum(-np.diff(q),0).sum()))


def selected(labels,rgb,max_per_axis=256):
    found=[]
    for axis in (0,1):
        lab=labels if axis==1 else labels.T;im=rgb if axis==1 else rgb.transpose(1,0,2)
        ys,xs=np.nonzero(lab[:,1:]!=lab[:,:-1]);parts=[]
        for y,x in zip(ys,xs):
            x+=1
            if y<3 or y>=lab.shape[0]-3 or x<14 or x>=lab.shape[1]-14:continue
            a,b=lab[y,x-1],lab[y,x]
            if not ((lab[y,x-12:x]==a).all() and (lab[y,x:x+12]==b).all()):continue
            if not ((lab[y-2:y+3,x-8:x-4]==a).all() and (lab[y-2:y+3,x+4:x+8]==b).all()):continue
            ref=im[y,x-12:x+12].astype(float)
            if np.max(np.abs(ref[:4].mean(0)-ref[-4:].mean(0)))<16:continue
            if parts and abs(y-parts[-1][0])<5 and abs(x-parts[-1][1])<5:continue
            parts.append((int(y),int(x)))
        if len(parts)>max_per_axis:parts=[parts[i] for i in np.linspace(0,len(parts)-1,max_per_axis).astype(int)]
        found.extend((axis,y,x) for y,x in parts)
    return found


def regression():
    p=np.zeros((24,3));p[12:]=100
    exact=measure(p,p);assert abs(exact['width10_90']-.8)<1e-8 and exact['rgb_off_reference_segment']==0
    blur=np.stack([np.convolve(np.pad(p[:,c],(2,2),mode='edge'),np.ones(5)/5,mode='valid') for c in range(3)],-1)
    assert measure(blur,p)['width10_90']>3.5
    assert measure(p*.1,p)['width10_90'] is None


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--previews',nargs='+',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    regression();data=m.load_data(a.data);all_records=[];hashes={}
    for row in data['validation']:
        target=m.t.r.read(row['target']);lab=m.load_region(row['region']);edges=selected(lab,target)
        images={'target':target}
        for folder in a.previews:
            path=Path(folder)/(row['id']+'_output.png');key=Path(folder).parent.name+'/'+Path(folder).name
            images[key]=m.t.r.read(path);hashes[str(path)]=file_hash(path)
        records=[]
        for axis,y,x in edges:
            ref=(target if axis==1 else target.transpose(1,0,2))[y,x-12:x+12].astype(float)
            records.append(dict(axis=axis,xy=[x,y],values={name:measure((im if axis==1 else im.transpose(1,0,2))[y,x-12:x+12].astype(float),ref) for name,im in images.items()}))
        all_records.append(dict(id=row['id'],edges=records))
    summary={}
    for key in all_records[0]['edges'][0]['values']:
        rr=[e['values'][key] for row in all_records for e in row['edges']];width=[r['width10_90'] for r in rr if r['width10_90'] is not None]
        summary[key]=dict(samples=len(rr),valid_widths=len(width),width_p50=float(np.median(width)),width_p90=float(np.quantile(width,.9)),
            median_contrast=float(np.median([r['relative_contrast'] for r in rr])),low_contrast_fraction=float(np.mean([r['relative_contrast']<=.25 for r in rr])),
            off_segment_rgb_rms_mean=float(np.mean([r['rgb_off_reference_segment'] for r in rr])))
    out=Path(a.out);out.mkdir(exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name)
    write_json(out/'results.json',dict(summary=summary,records=all_records,png_sha256=hashes,scope='Same straight target edges only, no thin structures or junction claim. PNG before codec.'))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
