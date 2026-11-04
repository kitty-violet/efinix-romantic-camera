"""Pixel-only revision. Original R1/R2 packages and user sources are read-only."""
from pathlib import Path
import sys,json,hashlib,shutil,cv2
import numpy as np
from PIL import Image

BASE=Path(r'E:\fpga\output\single4x-r1-style-library-20261102')
D=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'source'))
import oneblock as o
from oneblock import torch,F
R1=BASE/'models/PIXEL_R39'
R2=o.BASE/'revisions/film-pixel-r2/models/PIXEL_R39'
PHOTO=Path('C:/Users/Lenovo/AppData/Local/Temp/codex-clipboard-3f071f10-623c-429b-b354-1d728974ff3f.jpg')
VIDEO=Path('E:/电脑管家迁移文件/QQ聊天记录搬家/Tencent Files/2412479271/nt_qq/nt_data/Video/2026-11/Ori/228240feb77657a89091dafd461a431d.mp4')
ROI=(0,136,584,440)

def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def guard():
    assert shutil.disk_usage(D).free>7*2**30,'E free below 7GiB'
    assert sum(p.stat().st_size for p in D.rglob('*') if p.is_file())<2*2**30,'revision above 2GiB'

def fit720(a):
    h,w=a.shape[:2];scale=min(1280/w,720/h)
    nw=max(4,int(round(w*scale/4))*4);nh=max(4,int(round(h*scale/4))*4)
    interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR
    resized=cv2.resize(a,(nw,nh),interpolation=interpolation)
    x=((1280-nw)//8)*4;y=((720-nh)//8)*4
    out=np.zeros((720,1280,3),np.uint8);out[y:y+nh,x:x+nw]=resized
    return out,dict(x=x,y=y,width=nw,height=nh,source_width=w,source_height=h,resize='AREA' if scale<1 else 'LINEAR',round_dimensions_to_multiple_4=True)

def video_input(frame):
    x,y,w,h=ROI;return fit720(frame[y:y+h,x:x+w])

def model():
    ck=torch.load(R1/'selected.pt',map_location='cpu',weights_only=True)
    m=o.Model().cuda();m.load_state_dict(ck['model']);m.residual=ck['residual'];m.qat=True;m.leak=0
    return m

def rgb(t):return np.ascontiguousarray((t[0].permute(1,2,0).cpu().numpy()*128+128).round().clip(0,255).astype(np.uint8))

def block_metrics(a,source):
    c=a.reshape(180,4,320,4,3);assert np.all(c.min((1,3))==c.max((1,3)))
    z=a[::4,::4].astype(np.float32)
    inp=cv2.resize(source,(320,180),interpolation=cv2.INTER_AREA).astype(np.float32)
    mask=inp.max(2)>0
    # A colour island is counted only if every cardinal neighbour differs.
    ds=[np.max(np.abs(z[1:-1,1:-1]-q),2) for q in [z[:-2,1:-1],z[2:,1:-1],z[1:-1,:-2],z[1:-1,2:]]]
    island=(np.minimum.reduce(ds)>20)&mask[1:-1,1:-1]
    jumps=[];unsupported=[];signal=[]
    for axis in [0,1]:
        da=np.max(np.abs(np.diff(z,axis=axis)),2);di=np.max(np.abs(np.diff(inp,axis=axis)),2)
        valid=(mask[1:]&mask[:-1]) if axis==0 else (mask[:,1:]&mask[:,:-1])
        flat=valid&(di<4);jumps.extend(da[flat].tolist());unsupported.append(int(((da>24)&flat).sum()));signal.append(float(da[valid].mean()))
    return dict(unequal_4x4_cells=0,isolated_cells=int(island.sum()),unsupported_jumps=sum(unsupported),same_region_mean_jump=float(np.mean(jumps)) if jumps else 0.,all_grid_mean_jump=float(np.mean(signal)),unique_colours=len(np.unique(a[::4,::4].reshape(-1,3),axis=0)))

def prepare():
    o.setup(20261102);guard();(D/'inputs').mkdir(exist_ok=True)
    a=np.asarray(Image.open(PHOTO).convert('RGB'));p,meta=fit720(a);Image.fromarray(p).save(D/'inputs/photo.png')
    cap=cv2.VideoCapture(str(VIDEO));records=[];n=0
    while True:
        ok,b=cap.read()
        if not ok:break
        a=cv2.cvtColor(b,cv2.COLOR_BGR2RGB);p,vm=video_input(a)
        records.append(dict(index=n,pts_90000=round(cap.get(cv2.CAP_PROP_POS_MSEC)*90),original_rgb_sha256=digest(a),input720_sha256=digest(p)))
        if n in [0,536,1072]:Image.fromarray(p).save(D/'inputs'/f'video_{n:04d}.png')
        n+=1
    cap.release();assert n==1073
    assert all(a['pts_90000']<b['pts_90000'] for a,b in zip(records,records[1:]))
    save(D/'source-manifest.json',dict(photo=dict(path=str(PHOTO),sha256=sha(PHOTO),input720_sha256=digest(np.array(Image.open(D/'inputs/photo.png'))),placement=meta),video=dict(path=str(VIDEO),sha256=sha(VIDEO),frames=n,screen_recording=True,source_roi_xywh=ROI,roi_reason='Fixed entire mountain media viewport; phone status/header/comments excluded, source overlays retained',placement=vm,records=records),no_source_used_for_training=True))
    print('SOURCE_PIXELS_PREPARED',n,flush=True)

if __name__=='__main__':prepare()
