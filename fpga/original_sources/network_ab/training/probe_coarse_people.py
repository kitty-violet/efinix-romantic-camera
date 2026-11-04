"""Audit existing character deformation before choosing a coarsening operator."""
import json
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
from .publish_block_video import setup
from .fpga_video_resize import area_3to2
from .train_native_contours import tensor
from .train_b1_distillation import to_rgb
from .runtime import write_json,file_hash

OUT=Path(r'E:\fpga\model-training-workspace\runs\region-detail-deploy-20261004\coarse-people-20261005-probe')
WEIGHT=Path(r'E:\fpga\model-training-workspace\cache\torch\hub\checkpoints\lraspp_mobilenet_v3_large-d234d4ea.pth')


@torch.inference_mode()
def main():
    cfg,net,b,d,p,adapter,clip,state=setup();OUT.mkdir(exist_ok=False)
    assert WEIGHT.is_file()
    seg=lraspp_mobilenet_v3_large(weights=None,weights_backbone=None,num_classes=21)
    seg.load_state_dict(torch.load(WEIGHT,map_location='cpu',weights_only=True));seg=seg.cuda().eval()
    cap=cv2.VideoCapture(clip['clip'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);records=[]
    for i in range(450):
        ok,frame=cap.read();assert ok
        if i not in (90,210,330,449):continue
        source=area_3to2(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));pred=to_rgb(net(tensor(source)[None].cuda())[0])
        resized=cv2.resize(source,(768,432),interpolation=cv2.INTER_AREA)
        x=torch.from_numpy(resized.copy()).permute(2,0,1)[None].cuda().float()/255
        mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
        prob=seg((x-mean)/std)['out'].softmax(1)[0,15].cpu().numpy()
        mask=cv2.resize(prob,(1280,720),interpolation=cv2.INTER_LINEAR)
        old=np.array(Image.open(OUT.parent/'block-separation-video-20261004-v1'/f'frame_{i:03d}_native720.png'))
        for name,a in [('source',source),('student',pred),('old',old),('mask',np.uint8(mask*255))]:Image.fromarray(a).save(OUT/f'{i:03d}_{name}.png')
        np.save(OUT/f'{i:03d}_person_probability.npy',mask)
        records.append(dict(frame=i,person_fraction=float((mask>.5).mean())))
        print('PERSON_PROBE',i,flush=True)
    cap.release()
    f=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    cases=[(210,(120,435,440,715)),(330,(630,280,950,560)),(330,(0,430,320,710))]
    board=Image.new('RGB',(1280,316*len(cases)),'#17202a');draw=ImageDraw.Draw(board)
    for row,(i,box) in enumerate(cases):
        for col,(name,title) in enumerate([('source','原片'),('student','神经RGB预测'),('old','上一版最终'),('mask','自动人物概率')]):
            draw.text((col*320+4,row*316+4),title,font=f,fill='white')
            a=Image.open(OUT/f'{i:03d}_{name}.png').convert('RGB').crop(box);board.paste(a,(col*320,row*316+34))
    board.save(OUT/'deformation_audit.png')
    write_json(OUT/'audit.json',dict(records=records,source=clip['clip'],source_sha256=clip['clip_sha256'],segmenter=str(WEIGHT),segmenter_sha256=file_hash(WEIGHT),person_class=15,downloads=False))


if __name__=='__main__':main()
