from core import *

def main():
    d=OUT/'pixel-r28-video-review';rows=[]
    for scene in ['corridor','outdoor']:
        p=d/f'{scene}_native.mp4';cap=cv2.VideoCapture(str(p));hashes=[];sizes=set()
        while True:
            ok,bgr=cap.read()
            if not ok:break
            sizes.add((bgr.shape[1],bgr.shape[0]));hashes.append(hashlib.sha256(bgr.tobytes()).hexdigest())
            if len(hashes)==1:Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).save(d/f'{scene}_decoded_first.png')
        cap.release();assert len(hashes)==91 and len(set(hashes))==91 and sizes=={(1280,720)}
        rows.append(dict(scene=scene,path=str(p),frames=91,unique_decoded_frames=91,resolution=[1280,720],sha256=sha(p)))
    save_json(d/'encoded_verification.json',dict(clips=rows,subjective_motion_player_review=False,
        verification='Full decoded frame count/dimensions/uniqueness. Native RGB temporal metrics and original PTS in summary.json.'))
    print('R28_ENCODED_VIDEO_PASS',flush=True)

if __name__=='__main__':main()
