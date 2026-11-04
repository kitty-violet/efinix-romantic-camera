"""Read-only model diagnosis: highlight headroom and retained photographic detail."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
import torch
from .coherent_shared_model import SharedAppearanceNet
from .runtime import configure_workspace,file_hash,write_json


def luma(x):return x @ np.array([.2126,.7152,.0722],np.float32)


def texture_stats(a,b):
    ga,gb=luma(a),luma(b)
    rows={}
    for sigma in (1.,3.,6.):
        ha=ga-cv2.GaussianBlur(ga,(0,0),sigma)
        hb=gb-cv2.GaussianBlur(gb,(0,0),sigma)
        # Exclude the crop boundary introduced by the diagnostic filter.
        ha,hb=ha[20:-20,20:-20],hb[20:-20,20:-20]
        rows[str(sigma)]={'source_std':float(ha.std()),'output_std':float(hb.std()),
          'correlation':float(np.corrcoef(ha.ravel(),hb.ravel())[0,1]),
          'gain':float((ha*hb).sum()/(np.square(ha).sum()+1e-8))}
    design=np.concatenate([a.reshape(-1,3),np.ones((a.shape[0]*a.shape[1],1))],1)
    coeff=np.linalg.lstsq(design,b.reshape(-1,3),rcond=None)[0]
    pred=design@coeff
    denominator=np.square(b.reshape(-1,3)-b.mean((0,1))).sum()
    return {'bands':rows,'local_affine_color_r2':float(1-np.square(pred-b.reshape(-1,3)).sum()/max(denominator,1e-8))}


def main():
    p=argparse.ArgumentParser();p.add_argument('--render',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    configure_workspace();torch.set_num_threads(4);cv2.setNumThreads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=json.loads(args.render.read_text(encoding='utf-8'))
    state=torch.load(data['weights'],map_location='cpu',weights_only=True)
    if file_hash(data['weights'])!=data['weights_sha256']:raise ValueError('Weight drift')
    net=SharedAppearanceNet(**state['config']).cuda().eval();net.load_state_dict(state['generator'])
    args.output.mkdir(exist_ok=True,parents=True)
    heads=[]
    handle=net.head.register_forward_hook(lambda mod,inp,out:heads.append(out.detach().cpu().numpy()))
    # Fixed image coordinates are for measurements ONLY, never training or inference logic.
    rois={'test_street':{'sky':(400,20,500,140),'distant_trees':(450,260,690,365),
                        'road':(300,490,680,690),'near_foliage':(25,30,200,205)},
          'holdout_corridor':{'stone_steps':(30,420,360,700),'wall':(1010,280,1240,485)},
          'train_escalator':{'wall_tiles':(1010,280,1230,510)}}
    records=[];patches=[]
    for row in data['clips']:
        cap=cv2.VideoCapture(row['clip']);cap.set(cv2.CAP_PROP_POS_FRAMES,75)
        ok,bgr=cap.read();cap.release()
        if not ok:raise ValueError('Cannot read diagnostic frame')
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        a=np.array(Image.fromarray(rgb).resize((1280,720),Image.Resampling.BICUBIC)).astype(np.float32)
        x=torch.from_numpy(a.copy()).permute(2,0,1).unsqueeze(0).cuda()/128-1
        with torch.no_grad():out=net(x)
        b=out[0].permute(1,2,0).cpu().numpy()*128+128
        head=heads.pop()[0].transpose(1,2,0);unit=np.tanh(head)
        floor=np.maximum(0,a-state['config']['residual_scale']*128)
        high=a>245
        entry={'id':row['id'],'highlight_channel_fraction':float(high.mean()),
          'highlight_channels_within_4_levels_of_floor':float(((b-floor)<4)[high].mean()) if high.any() else None,
          'negative_head_saturation_fraction':float((unit<-.98).mean()),'rois':{}}
        for label,(x1,y1,x2,y2) in rois.get(row['id'],{}).items():
            aa,bb=a[y1:y2,x1:x2],b[y1:y2,x1:x2]
            uu=unit[y1:y2,x1:x2];ff=floor[y1:y2,x1:x2]
            entry['rois'][label]={'xyxy_720':[x1,y1,x2,y2],
              'source_luma_mean':float(luma(aa).mean()),'output_luma_mean':float(luma(bb).mean()),
              'source_rgb_mean':aa.mean((0,1)).tolist(),'output_rgb_mean':bb.mean((0,1)).tolist(),
              'within_4_levels_of_floor':float(((bb-ff)<4).mean()),
              'negative_saturation_fraction':float((uu<-.98).mean()),
              'texture':texture_stats(aa,bb)}
            patches.append((row['id']+'/'+label,aa,bb))
        records.append(entry)
    handle.remove()
    board=Image.new('RGB',(1200,240*len(patches)),'#151515');draw=ImageDraw.Draw(board)
    for i,(label,aa,bb) in enumerate(patches):
        for j,im in enumerate((aa,bb)):
            tile=Image.fromarray(im.round().clip(0,255).astype(np.uint8))
            tile.thumbnail((590,210),Image.Resampling.LANCZOS)
            board.paste(tile,(j*600,i*240+25))
            draw.text((j*600+4,i*240+4),label+(' SOURCE' if j==0 else 'CURRENT MODEL'),fill='white')
    board.save(args.output/'measured_regions.jpg',quality=96)
    report={'status':'DIAGNOSTIC_NOT_A_NEW_TRAINED_RESULT','weights_sha256':data['weights_sha256'],
      'source_render_sha256':file_hash(args.render),'frame':75,
      'analytical_head_max_reduction_8bit':state['config']['residual_scale']*128,
      'analytical_white255_minimum_output':255-state['config']['residual_scale']*128,
      'clips':records,'notes':['ROIs are diagnostic only, not semantic segmentation or per-scene training.',
      'High-pass correlation describes retained spatial pattern, not perceptual style acceptance.',
      'Saturated source highlights cannot recover real sky detail by changing the output parameterization.']}
    write_json(args.output/'diagnosis.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
