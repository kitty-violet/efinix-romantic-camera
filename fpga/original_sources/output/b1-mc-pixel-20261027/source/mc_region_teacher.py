"""RGB-region full-cube supervision, never an inference stage.

Small sharp bounded regions may acquire a coherent top+side. Broad surfaces,
soft illumination and long shadows stay planar. The runtime student receives
only RGB; all connected components and polygons here are offline targets.
"""
from core import *
from pixel_targets import panel
from r15_structure_guard_probe import fixtures

def low_rgb(a):
    z=cv2.GaussianBlur(a.astype(np.float32),(3,3),.65,borderType=cv2.BORDER_REPLICATE)
    h,w=z.shape[:2];return z.reshape(h//4,4,w//4,4,3).mean((1,3))

def palette_from(arrays,k=18):
    # Fixed training-only palette; no frame-dependent palette is required by B1.
    q=np.concatenate([low_rgb(a).reshape(-1,3)[::23] for a in arrays]).astype(np.float32)
    cv2.setRNGSeed(271173)
    transform=np.array([[.15,.85,-.425],[.30,0,.85],[.15,-.85,-.425]],np.float32)
    features=q@transform
    _,lab,_=cv2.kmeans(features,k,None,(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,50,.1),3,cv2.KMEANS_PP_CENTERS)
    centres=np.stack([q[lab[:,0]==j].mean(0) for j in range(k)])
    return centres[np.argsort(centres.mean(1))]

def teacher(a,palette,return_meta=False):
    low=low_rgb(a);h,w=low.shape[:2]
    distances=((low[:,:,None,:]-palette[None,None])**2).sum(3);labels=distances.argmin(2).astype(np.int32)
    # Label decisions are joint RGB. An isolated mixed colour is replaced by a
    # neighbouring existing palette colour, with one centre-heavy Potts sweep.
    original=labels.copy();scores=distances.copy()
    for j in range(len(palette)):
        count=cv2.boxFilter((original==j).astype(np.float32),-1,(3,3),normalize=False,borderType=cv2.BORDER_REPLICATE)
        scores[:,:,j]+=220*(9-count)
    labels=scores.argmin(2).astype(np.int32);base=palette[labels].copy()
    # Component proposals are bounded local fronts, not brightness-as-height.
    proposals=[]
    for j in range(len(palette)):
        n,lab,stats,_=cv2.connectedComponentsWithStats((labels==j).astype(np.uint8),connectivity=4)
        for cid in range(1,n):
            x,y,bw,bh,area=map(int,stats[cid]);fill=area/(bw*bh)
            if area<9 or min(bw,bh)<3 or max(bw,bh)>22 or fill<.60:continue
            if x<1 or y<1 or x+bw>=w-1 or y+bh>=h-1:continue
            # Median contrast across the ACTUAL outer rectangle rejects smooth
            # light gradients; no value/brightness directly becomes a height.
            edges=[]
            for axis,pos,lo,hi in [(0,y,x,x+bw),(0,y+bh,x,x+bw),(1,x,y,y+bh),(1,x+bw,y,y+bh)]:
                stack=[]
                for off in range(-2,3):
                    z=max(0,min((h if axis==0 else w)-2,pos+off))
                    a0=low[z,lo:hi] if axis==0 else low[lo:hi,z]
                    a1=low[z+1,lo:hi] if axis==0 else low[lo:hi,z+1]
                    stack.append(np.linalg.norm(a1-a0,axis=1))
                edges.extend(np.max(stack,axis=0).tolist())
            edge=float(np.median(edges))
            if edge<42:continue
            colour=np.median(low[y:y+bh,x:x+bw].reshape(-1,3),axis=0)
            proposals.append(dict(x=x,y=y,w=bw,h=bh,area=area,fill=fill,edge=edge,colour=colour))
    # A contained patch belongs to its enclosing front; it must not become a
    # second cube just because its paint/illumination differs.
    kept=[]
    for p in sorted(proposals,key=lambda p:-p['w']*p['h']):
        if any(p['x']>=q['x'] and p['y']>=q['y'] and p['x']+p['w']<=q['x']+q['w'] and p['y']+p['h']<=q['y']+q['h'] for q in kept):continue
        kept.append(p)
    canvas=Image.fromarray(np.clip(base+.5,0,255).astype(np.uint8));draw=ImageDraw.Draw(canvas)
    for p in sorted(kept,key=lambda p:p['y']+p['h']):
        x,y,bw,bh=p['x'],p['y'],p['w'],p['h'];d=min(6,max(2,min(bw,bh)//2));c=p['colour']
        top=tuple(np.clip(c*.82+255*.18,0,255).astype(int));side=tuple(np.clip(c*.68,0,255).astype(int));front=tuple(np.clip(c,0,255).astype(int))
        draw.polygon([(x,y),(x+bw,y),(x+bw+d,y-d),(x+d,y-d)],fill=top)
        draw.polygon([(x+bw,y),(x+bw,y+bh),(x+bw+d,y+bh-d),(x+bw+d,y-d)],fill=side)
        draw.rectangle((x,y,x+bw-1,y+bh-1),fill=front);p['depth']=d
    out=np.asarray(canvas).repeat(4,0).repeat(4,1)
    meta=dict(regions=len(kept),proposals=[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in p.items()} for p in kept],
        brightness_as_height=False,shared_RGB_and_depth_per_region=True,teacher_only=True)
    return (out,meta) if return_meta else out

def main():
    from train_pixel import data
    setup(271173);d=OUT/'r30-joint-colour-teacher';d.mkdir(exist_ok=True);arrays,vals=data();palette=palette_from(arrays)
    save_json(d/'palette.json',dict(colours=palette.tolist(),training_only=True,source_split_sha256=sha(OUT/'data_split.json')))
    cases=[('corridor',read(OUT/'data-quality/VID_20260924_231158_source.png'),None),('outdoor',read(OUT/'targets/outdoor_source.png'),None),*fixtures()];rows=[]
    for name,a,pos in cases:
        t,m=teacher(a,palette,True);Image.fromarray(t).save(d/f'{name}_target.png')
        panel([a,t],['原图','联合RGB区域 / 完整三面教师，非B1输出'],d/f'{name}_comparison.jpg',width=640)
        rows.append(dict(name=name,positive_control=pos,**m));print('REGION_TEACHER',name,m['regions'],flush=True)
    negatives=[r for r in rows if r['positive_control'] is False];positives=[r for r in rows if r['positive_control'] is True]
    good=all(r['regions']==0 for r in negatives) and all(r['regions']>=1 for r in positives)
    save_json(d/'preview.json',dict(records=rows,real_scene_style_accepted=False,inference_component_processing=False,fixture_gate_pass=good))
    assert good,'Do not train a teacher which fails basic full-cube/light controls'

if __name__=='__main__':main()
