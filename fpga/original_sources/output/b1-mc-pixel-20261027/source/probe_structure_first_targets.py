"""Training-target probes: simplify shared structures without inventing depth.

No model or inference pipeline uses these CPU filters. These probes compare
edge-sparse and axis-aligned region abstraction while retaining original light
and perspective. Neither is a 3D reconstruction nor accepted MC target yet.
"""
from core import *
from pixel_targets import panel

def l0_smooth(a,strength=.015,iterations=12):
    # Half-quadratic splitting for L0 gradient smoothing, padded to suppress
    # periodic FFT seams. This is an independent implementation of the method.
    pad=12;z=np.pad(a,((pad,pad),(pad,pad),(0,0)),mode='reflect').astype(np.float32)/255
    h,w=z.shape[:2];fy=np.fft.fftfreq(h)[:,None];fx=np.fft.fftfreq(w)[None,:]
    denom=(2-2*np.cos(2*np.pi*fx))+(2-2*np.cos(2*np.pi*fy));original=np.fft.fft2(z,axes=(0,1));s=z.copy();beta=2*strength
    for _ in range(iterations):
        dx=np.roll(s,-1,axis=1)-s;dy=np.roll(s,-1,axis=0)-s
        small=np.sum(dx*dx+dy*dy,2)<strength/beta;dx[small]=0;dy[small]=0
        div=np.roll(dx,1,axis=1)-dx+np.roll(dy,1,axis=0)-dy
        s=np.fft.ifft2((original+beta*np.fft.fft2(div,axes=(0,1)))/(1+beta*denom[...,None]),axes=(0,1)).real
        beta*=2
    return np.clip(s[pad:-pad,pad:-pad]*255,0,255)

def quadrant_flatten(a,radius=4):
    # Axis-aligned four-quadrant variance-weighted averaging. It avoids a hard
    # winner switch. Not a reproduction of anisotropic Kuwahara flow filtering.
    z=a.astype(np.float32);h,w=z.shape[:2];pad=np.pad(z,((radius,radius),(radius,radius),(0,0)),mode='edge')
    means=[];weights=[];k=radius+1
    for yy,xx in [(0,0),(0,radius),(radius,0),(radius,radius)]:
        mean=cv2.boxFilter(pad,-1,(k,k),anchor=(0,0),normalize=True,borderType=cv2.BORDER_REPLICATE)[yy:yy+h,xx:xx+w]
        square=cv2.boxFilter(pad*pad,-1,(k,k),anchor=(0,0),normalize=True,borderType=cv2.BORDER_REPLICATE)[yy:yy+h,xx:xx+w]
        variance=np.maximum(square-mean*mean,0).mean(2);weight=1/(1+(variance/100)**2)
        means.append(mean);weights.append(weight)
    weights=np.stack(weights);weights/=weights.sum(0,keepdims=True)
    return sum(m*w[...,None] for m,w in zip(means,weights))

def target(a,method):
    z=cv2.GaussianBlur(a.astype(np.float32),(3,3),0,borderType=cv2.BORDER_REPLICATE)
    h,w=z.shape[:2];low=z.reshape(h//4,4,w//4,4,3).mean((1,3))
    if method=='edge-sparse':out=l0_smooth(low,.018)
    else:out=quadrant_flatten(low,4)
    # Keep a little original large-scale shading instead of assigning relief
    # height to either luminance or quantisation levels. No displaced samples.
    base=cv2.GaussianBlur(low,(0,0),2,borderType=cv2.BORDER_REPLICATE)
    smoothed=cv2.GaussianBlur(out,(0,0),2,borderType=cv2.BORDER_REPLICATE)
    out=out+.3*(base-smoothed)
    return np.clip(out+.5,0,255).astype(np.uint8).repeat(4,0).repeat(4,1)

def main():
    d=OUT/'mc-structure-first-target-probe';d.mkdir(exist_ok=True);disk_guard();stats=[]
    for scene in ['outdoor','indoor']:
        a=read(OUT/f'targets/{scene}_source.png');images=[a]
        for method in ['edge-sparse','axis-regions']:
            o=target(a,method);Image.fromarray(o).save(d/f'{scene}_{method}.png');images.append(o)
            stats.append(dict(scene=scene,method=method,input_geometry_displaced=False,added_height_or_facets=False))
        panel(images,['原图','结构稀疏化目标 / 非B1','邻域块面归并目标 / 非B1'],d/f'{scene}_comparison.jpg')
    save_json(d/'probe.json',dict(records=stats,teacher_only=True,MC_accepted=False,student_trained=False,
        choice='assess global arrangement and retained source 3D before any further student fit',
        sources=['https://www.cse.cuhk.edu.hk/~leojia/projects/L0smoothing/',
                 'https://www.kyprianidis.com/p/pg2009/jkyprian-pg2009.pdf']))
    print('STRUCTURE_FIRST_TARGET_PROBE_COMPLETE',flush=True)

if __name__=='__main__':main()
