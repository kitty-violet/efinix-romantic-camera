"""Deterministic image-space study on actual learned RGB predictions.

No teacher inference, new neural weights, material labels or random brush texture.
Inspired by edge-aware abstraction and soft quantization, not an exact paper port.
"""
import cv2
import numpy as np

PRESETS={
    'tone':dict(label='A 仅色彩与明暗',kind='tone'),
    'poster':dict(label='B 哑光色块',kind='poster',method='domain',spatial=48,range=.22,
                  l_step=8.,ab_step=8.,q_sharp=4.,q_mix=.78,
                  edge_sigma=1.35,edge_threshold=.75,edge_scale=2.6,edge_darkening=10.),
    'poster_coarse':dict(label='C 较粗色块',kind='poster',method='meanshift',median=5,spatial=8,range=20,max_level=1,
                  l_step=10.,ab_step=10.,q_sharp=5.,q_mix=.88,
                  edge_sigma=1.65,edge_threshold=.85,edge_scale=2.8,edge_darkening=12.),
    'watercolor':dict(label='D OpenCV绘画化对照',kind='stylization',spatial=70,range=.35),
}
TONE=dict(chroma_gain=1.20,cool_b=1.15,green_a=.5,warm_b_reduction=.055,
          black_l=.01,contrast=1.065,highlight_shoulder=.30)


def as_lab(rgb):
    return cv2.cvtColor(rgb.astype(np.float32)/255,cv2.COLOR_RGB2Lab)


def as_rgb(lab):
    return np.uint8(np.round(np.clip(cv2.cvtColor(lab.astype(np.float32),cv2.COLOR_Lab2RGB),0,1)*255))


def tone(rgb,adapter):
    lab=as_lab(rgb)
    chroma=((lab/100)@adapter[:3]+adapter[3])*100*TONE['chroma_gain']
    gate=np.clip(np.linalg.norm(chroma,axis=2)/12,0,1)
    chroma[:,:,0]-=TONE['green_a']*gate
    chroma[:,:,1]-=TONE['cool_b']*gate+TONE['warm_b_reduction']*np.maximum(chroma[:,:,1],0)
    l=lab[:,:,0]/100
    l=np.clip((l-TONE['black_l'])/(1-TONE['black_l']),0,1)
    # Fixed monotone contrast curve, then a smooth highlight shoulder.
    l=np.clip(.5+TONE['contrast']*(l-.5),0,1)
    l-=TONE['highlight_shoulder']*np.maximum(l-.78,0)**2/.22
    lab[:,:,0]=l*100;lab[:,:,1:]=chroma
    return as_rgb(lab)


def soft_levels(x,step,sharpness,mix):
    # Continuous across bin boundaries; contrast flattens near color levels.
    u=x/step;n=np.floor(u);f=u-n
    q=n+.5+.5*np.tanh(sharpness*(f-.5))/np.tanh(sharpness*.5)
    return ((1-mix)*u+mix*q)*step


def transform(rgb,adapter,preset='poster',stages=False):
    cfg=PRESETS[preset];colored=tone(rgb,adapter);details={'tone':colored}
    if cfg['kind']=='tone':out=colored
    elif cfg['kind']=='stylization':
        bgr=cv2.cvtColor(colored,cv2.COLOR_RGB2BGR)
        out=cv2.cvtColor(cv2.stylization(bgr,sigma_s=cfg['spatial'],sigma_r=cfg['range']),cv2.COLOR_BGR2RGB)
    else:
        bgr=cv2.cvtColor(colored,cv2.COLOR_RGB2BGR)
        if cfg['method']=='domain':
            filtered=cv2.edgePreservingFilter(bgr,flags=cv2.RECURS_FILTER,
                                             sigma_s=cfg['spatial'],sigma_r=cfg['range'])
        else:
            # Median removes small high-contrast flecks before mode-seeking color merging.
            cleaned=cv2.medianBlur(bgr,cfg['median'])
            filtered=cv2.pyrMeanShiftFiltering(cleaned,sp=cfg['spatial'],sr=cfg['range'],maxLevel=cfg['max_level'])
        base=cv2.cvtColor(filtered,cv2.COLOR_BGR2RGB);lab=as_lab(base);l=lab[:,:,0].copy()
        details['surface']=base
        lab[:,:,0]=soft_levels(l,cfg['l_step'],cfg['q_sharp'],cfg['q_mix'])
        for channel in [1,2]:
            lab[:,:,channel]=soft_levels(lab[:,:,channel],cfg['ab_step'],cfg['q_sharp'],cfg['q_mix']*.40)
        details['blocks']=as_rgb(lab)
        narrow=cv2.GaussianBlur(l,(0,0),cfg['edge_sigma'])
        broad=cv2.GaussianBlur(l,(0,0),cfg['edge_sigma']*1.8)
        response=np.maximum(broad-narrow-cfg['edge_threshold'],0)/cfg['edge_scale']
        line=1-np.exp(-response**2)
        # Only dark-side emphasis of surviving coarse structures, no white halo.
        lab[:,:,0]=np.clip(lab[:,:,0]-cfg['edge_darkening']*line,0,100)
        details['edges']=np.uint8(np.round(255*(1-line)))
        out=as_rgb(lab)
    if stages:return out,details
    return out
