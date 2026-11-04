"""CPU audit of saved contour maps; no output image enhancement is applied."""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from .runtime import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);a=p.parse_args();rows=[]
    for path in sorted(a.folder.glob('*_teacher_contour.png')):
        name=path.name.removesuffix('_teacher_contour.png')
        t=cv2.imread(str(path),0)[::2,::2]>30
        g=cv2.imread(str(a.folder/(name+'_predicted_contour.png')),0)[::2,::2]
        t=t[24:-24,24:-24];g=g[24:-24,24:-24]
        dilated=cv2.dilate(t.astype(np.uint8),np.ones((3,3),np.uint8))>0
        distance=cv2.distanceTransform((~t).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        trials=[]
        for threshold in (16,32,64,96,128,160,192):
            predicted=g>=threshold;pd=cv2.dilate(predicted.astype(np.uint8),np.ones((3,3),np.uint8))>0
            precision=float((predicted&dilated).sum()/max(1,predicted.sum()));recall=float((t&pd).sum()/max(1,t.sum()))
            trials.append(dict(threshold=threshold,precision_within_one360p_pixel=precision,recall_within_one360p_pixel=recall,
                f1=2*precision*recall/max(1e-9,precision+recall),predicted_area_over_target=float(predicted.sum()/max(1,t.sum())),
                predicted_mean_distance360p=float(distance[predicted].mean()) if predicted.any() else None))
        rows.append(dict(id=name,teacher_fraction=float(t.mean()),trials=trials))
    write_json(a.folder/'contour_localization.json',dict(records=rows,threshold_sweep_not_inference_postprocessing=True,
        units='360p pixels; one equals two native720 pixels',not_blind=True,metric_is_not_visual_or_temporal_acceptance=True))
    print(json.dumps([dict(id=r['id'],best=max(r['trials'],key=lambda t:t['f1'])) for r in rows]),flush=True)


if __name__=='__main__':main()
