"""Measure compression error and motion-compensated changes; no quality pass claim."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from .runtime import configure_workspace, write_json, file_hash


def main():
    root=configure_workspace();run=root/'runs/wavelet-game-probe-20261003'
    report=json.loads((run/'video/report.json').read_text());cv2.setNumThreads(4)
    records=[]
    for row in report['records']:
        name=row['source']['id']
        old=root/'runs/coherent-surface-20261002/final-4850'/f'{name}_styled_1080p.mp4'
        paths=[row['source']['clip'],row['output'],str(old)]
        caps=[cv2.VideoCapture(p) for p in paths]
        previous=None;values=[];encoding=[];n=0
        while True:
            reads=[c.read() for c in caps]
            if not any(ok for ok,_ in reads):break
            if not all(ok for ok,_ in reads):raise RuntimeError('Frame count mismatch')
            frames=[cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB) for _,bgr in reads]
            if n in (0,75,149):
                raw=np.asarray(Image.open(run/'video'/f'{name}_{n:03d}_preencode1080.png')).astype(float)
                mse=np.square(raw-frames[1].astype(float)).mean()
                encoding.append({'frame':n,'psnr_db':float(10*np.log10(255**2/max(mse,1e-12))),
                    'rgb_mae':float(np.abs(raw-frames[1]).mean())})
            small=[cv2.resize(im,(640,360),interpolation=cv2.INTER_AREA).astype(np.float32) for im in frames]
            if previous is not None:
                curgray=cv2.cvtColor(small[0].astype(np.uint8),cv2.COLOR_RGB2GRAY)
                prevgray=cv2.cvtColor(previous[0].astype(np.uint8),cv2.COLOR_RGB2GRAY)
                flow=cv2.calcOpticalFlowFarneback(curgray,prevgray,None,.5,3,21,3,5,1.2,0)
                yy,xx=np.mgrid[:360,:640].astype(np.float32)
                mx,my=xx+flow[:,:,0],yy+flow[:,:,1]
                warped=[cv2.remap(im,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT) for im in previous]
                mask=(np.abs(small[0]-warped[0]).mean(2)<8)&(mx>=2)&(mx<638)&(my>=2)&(my<358)
                if mask.sum()<1000:raise RuntimeError('Too few reliable flow pixels')
                measurements={}
                for i,k in ((1,'wavelet'),(2,'stable4850')):
                    delta_now=small[i]-small[0];delta_prev=warped[i]-warped[0]
                    measurements[k]=float(np.abs(delta_now-delta_prev).mean(2)[mask].mean())
                values.append({'frame':n,'valid_fraction':float(mask.mean()),**measurements})
            previous=small;n+=1
        for c in caps:c.release()
        if n!=150:raise RuntimeError('Expected 150 frames')
        records.append({'clip':name,'frames':n,'encoding_error':encoding,
            'motion_style_delta':{k:{'median':float(np.median([v[k] for v in values])),
                                   'p95':float(np.percentile([v[k] for v in values],95)),
                                   'worst_frame':max(values,key=lambda v:v[k])['frame']}
                                   for k in ('wavelet','stable4850')},'all_pairs':values})
        print(name,records[-1]['motion_style_delta'],encoding,flush=True)
    write_json(run/'video/quality_review.json',{'status':'MEASURED_NOT_PERCEPTUAL_ACCEPTANCE',
        'records':records,'method':'640x360 source backward Farneback flow; in-bounds source residual<8; RGB temporal style delta, 0..255',
        'limits':'Flow can be inaccurate; metric depends on style strength; lower is not proof of no flicker. Compare actual videos.',
        'code_sha256':file_hash(__file__)})


if __name__=='__main__':main()
