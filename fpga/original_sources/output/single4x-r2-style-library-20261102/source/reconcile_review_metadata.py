"""Backfill diagnostics omitted by the earlier four-style renderer schema.

Only measured raw hashes and saved raw PNG pixels are used. No network or
video output is changed, and missing evidence is never assumed to pass.
"""
import json,hashlib
import numpy as np
from PIL import Image
from library import D,save_json
from render_review import tiny_islands,image_metrics

def digest(a):return hashlib.sha256(a.tobytes()).hexdigest()

def main():
    changed=[]
    for key in ['A1','INK','OLD_FILM','PIXEL_R39']:
        static_path=D/'review/static'/key/'verification.json'
        static=json.loads(static_path.read_text('utf-8'))
        for r in static['records']:
            scene=r['scene'];path=D/'review/native'/key/(scene+'.json')
            report=json.loads(path.read_text('utf-8'));count=report['frames']
            new_hashes=report['new_raw_sha256'];old_hashes=report['old_raw_sha256']
            assert len(new_hashes)==len(old_hashes)==count==152
            assert all(isinstance(h,str) and len(h)==64 for h in new_hashes+old_hashes)
            assert report['native_video']['unique_encoded_frames']==count
            report['unique_new_raw_frames']=len(set(new_hashes))
            report['exact_raw_metric_frames']=count
            assert report['unique_new_raw_frames']==count
            report['raw_metric_scope']='all source frames; old and new BIN executed per frame'
            n=count//2;pics={}
            for tag,hashes in [('old',old_hashes),('new',new_hashes)]:
                p=D/'review/stills'/key/f'{scene}_{n:03d}_{tag}.png'
                a=np.array(Image.open(p).convert('RGB'));assert digest(a)==hashes[n]
                r['metrics'][tag+'_tiny_islands']=tiny_islands(a);pics[tag]=a
            assert abs(image_metrics(pics['old'],pics['new'])['mae']-r['metrics']['mae'])<1e-6
            report['schema_diagnostic_backfill']='raw hash counts and 4x4 component counts from SHA-verified existing PNGs; pixels unchanged'
            save_json(path,report);changed.append([key,scene])
        static['tiny_islands_source']='native midpoint PNGs verified against per-frame integer output SHA256'
        save_json(static_path,static)
    save_json(D/'review/metadata-backfill.json',dict(status='PASS',records=changed,video_pixels_modified=False,coefficients_modified=False))
    print('REVIEW_METADATA_BACKFILL_COMPLETE',len(changed),flush=True)

if __name__=='__main__':main()
