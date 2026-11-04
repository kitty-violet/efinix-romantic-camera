"""Decode delivered native videos and check unique frames and dimensions."""
from pathlib import Path
import cv2,json,hashlib
from PIL import Image

def main():
    root=Path(__file__).resolve().parents[1]/'pixel-a-shapes-and-mc-review';out=root/'encoded-video-qa';out.mkdir(exist_ok=True);rows=[]
    for scene,expected in [('station',90),('outdoor',91)]:
        path=root/f'{scene}_pixel_r23.mp4';cap=cv2.VideoCapture(str(path));hashes=[];sizes=set();index=0
        while True:
            ok,bgr=cap.read()
            if not ok:break
            sizes.add((bgr.shape[1],bgr.shape[0]));hashes.append(hashlib.sha256(bgr.tobytes()).hexdigest())
            if index in [0,30,60]:Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).save(out/f'{scene}_decoded_{index:04d}.png')
            index+=1
        cap.release();assert index==expected and sizes=={(1280,720)} and len(set(hashes))==index
        rows.append(dict(scene=scene,path=str(path),frames=index,decoded_unique_frames=len(set(hashes)),size=[1280,720]))
    path=root/'station_ceiling_crop.mp4';cap=cv2.VideoCapture(str(path));ok,bgr=cap.read();cap.release()
    assert ok and bgr.shape[:2]==(356,1536)
    Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).save(out/'ceiling_decoded_0000.png')
    (out/'verification.json').write_text(json.dumps(dict(native_videos=rows,crop_dimensions=[1536,356],
        subjective_motion_player_review=False,verification='full decode, native dimensions, every encoded frame unique; see raw-RGB/PTS reports for numerical temporal analysis'),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('ENCODED_NATIVE_VIDEO_CONTENT_PASS',json.dumps(rows),flush=True)

if __name__=='__main__':main()
