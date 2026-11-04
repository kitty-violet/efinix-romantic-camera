"""Two existing B1 residual stages learn coarse structural boundaries."""
import argparse
import copy
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from . import train_b1_structural_gate as gate
from .train_b1_structural_features import FeatureStudent,FeatureTraining
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible

OUT=gate.ROOT/'b1-learned-deep-contour-20261005-v4'


def labeler(rgb):
    # Do not round the smoothed image to uint8: low-gradient ridges break into
    # tiny Canny components and are then incorrectly removed by size filtering.
    gray=cv2.GaussianBlur(cv2.cvtColor(rgb.astype(np.float32),cv2.COLOR_RGB2GRAY),(0,0),5.)
    dx=np.rint(cv2.Sobel(gray,cv2.CV_32F,1,0)*16).astype(np.int16)
    dy=np.rint(cv2.Sobel(gray,cv2.CV_32F,0,1)*16).astype(np.int16)
    edge=cv2.Canny(dx,dy,5*16,10*16,L2gradient=True)
    n,labels,stats,_=cv2.connectedComponentsWithStats(edge,8);keep=np.zeros(n,np.uint8)
    for k in range(1,n):
        if stats[k,4]>=48 and max(stats[k,2],stats[k,3])>=20:keep[k]=255
    thin=keep[labels];distance=cv2.distanceTransform(255-thin,cv2.DIST_L2,5)
    return thin,np.maximum(0,1-distance/5).astype(np.float32)


class DeepStudent(FeatureStudent):
    first_gate_node='res3_conv1.dw'
    rectified_gate_node='res3_add'

    def __init__(self):
        super().__init__()
        with torch.no_grad():
            # Move the surface update into the third existing residual. Stage
            # one now refines hidden features; stage two predicts the barrier.
            for suffix in ['conv1__dw','conv1__pw','conv2__dw','conv2__pw']:
                for dest,src in [('res3_'+suffix,'res2_'+suffix),('res2_'+suffix,'res1_'+suffix)]:
                    self.weight[dest].copy_(self.weight[src]);self.bias[dest].copy_(self.bias[src])
                    getattr(self,dest+'_m').copy_(getattr(self,src+'_m'))
                    self.package['parameters'][dest.replace('__','.')]['multipliers']=getattr(self,dest+'_m').flatten().to(torch.int64).tolist()
            self.weight['res1_conv2__pw'].zero_();self.bias['res1_conv2__pw'].zero_()
            for kind,params in [('w',self.weight),('b',self.bias)]:
                for key,value in params.items():
                    if not key.startswith(('res1_','res2_')):continue
                    value.requires_grad_(True);mask=torch.zeros_like(value)
                    if key=='res2_conv2__pw':
                        if kind=='w':mask[24:32,6:24]=1
                        else:mask[24:32]=1
                    elif key.endswith('__pw') and kind=='w':mask[6:24,6:24]=1
                    else:mask[6:24]=1
                    for ending,tensor in [('mask',mask),('initial',value.detach().clone())]:
                        name=kind+'_'+key+'_'+ending
                        if hasattr(self,name):getattr(self,name).copy_(tensor)
                        else:self.register_buffer(name,tensor)
                    self.masks[(kind,key)]=True

    def project(self):
        super().project()
        with torch.no_grad():
            for params in (self.weight,self.bias):
                value=params['res2_conv2__pw'];value[25:32].copy_(value[24:25].expand_as(value[25:32]))

    def forward(self,x,all_nodes=False,full=False):
        nodes=super().forward(x,all_nodes=True,full=(full or all_nodes),stop_node='res2_add')
        return nodes if all_nodes else nodes['res2_add'][:,24:25]


class DeepTraining(FeatureTraining):
    def __init__(self,data):
        super().__init__(data);self.net=DeepStudent().cuda().eval()
        self.opt=torch.optim.Adam([p for p in self.net.parameters() if p.requires_grad],lr=.025)


@torch.no_grad()
def fp32_review(tr,data,tag):
    dest=OUT/tag;dest.mkdir();previous=tr.net.quantized;tr.net.quantized=False
    for r in data:
        if r['split']=='train':continue
        rgb=np.array(Image.open(r['source']));x=torch.tensor(rgb.transpose(2,0,1)[None].copy(),device='cuda',dtype=torch.float32)/128-1
        p=tr.net(x).clamp_min(0)[0,0].cpu().numpy()
        Image.fromarray(np.clip(p*255/24,0,255).astype(np.uint8)).resize((1280,720),Image.Resampling.NEAREST).save(dest/(r['id']+'_gate.png'))
    tr.net.quantized=previous
    from PIL import ImageDraw,ImageFont
    board=Image.new('RGB',(1260,1256),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    for i,(key,box,label) in enumerate(gate.CROPS):
        row=next(r for r in data if r['id']==key)
        for j,(path,title) in enumerate([(row['source'],'原片 '+label),(OUT/'labels'/(key+'_thin.png'),'较粗结构监督'),(dest/(key+'_gate.png'),'小网实际 FP32 轮廓')]):
            board.paste(Image.open(path).convert('RGB').crop(box),(420*j,314*i+34));draw.text((420*j+5,314*i+4),title,font=font,fill='white')
    board.save(dest/'native_fp32_review.png')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','preflight','train']);a=ap.parse_args()
    configure_workspace();set_reproducible(2026100506);cv2.setNumThreads(2);contract_guard();gate.OUT=OUT
    if a.mode=='prepare':
        gate.prepare(labeler);write_json(OUT/'config.json',dict(steps=2000,fp32=True,
            label='Frozen target float32 Gaussian5, signed16 Sobel x16 Canny5/10 x16, component48 extent20 distance5',
            graph='Unchanged B1: res1/res2 learn contour, res3 performs the actual constrained update',
            acceptance='Inspect FP32 before any further QAT; labels and lowered density are not acceptance'))
        return
    data=json.loads((OUT/'data.json').read_text(encoding='utf-8'))
    for r in data:
        for key in ('source','target','label'):assert file_hash(r[key])==r[key+'_sha256']
    tr=DeepTraining(data)
    if a.mode=='preflight':
        dest=OUT/'preflight';dest.mkdir();native=tr.update(True);crop=tr.update();tr.save(dest/'resume.pt');ref=tr.update()
        saved={k:v.clone() for k,v in tr.net.state_dict().items()};tr.restore(dest/'resume.pt');got=tr.update()
        err=max(float((v-saved[k]).abs().max()) for k,v in tr.net.state_dict().items());assert err<1e-6 and ref['crop']==got['crop']
        gate.evaluate(tr,data,'preflight-export')
        write_json(dest/'result.json',dict(status='PASS',native=native,crop=crop,resume_error=err,
            code_sha256=file_hash(__file__),gate_code_sha256=file_hash(gate.__file__)))
        print('DEEP_PREFLIGHT_PASS',err,flush=True);return
    pre=json.loads((OUT/'preflight/result.json').read_text());assert pre['status']=='PASS' and pre['code_sha256']==file_hash(__file__) and pre['gate_code_sha256']==file_hash(gate.__file__)
    for p in [__file__,gate.__file__,Path(__file__).with_name('train_b1_structural_features.py')]:shutil.copy2(p,OUT/Path(p).name)
    with (OUT/'metrics.jsonl').open('x',encoding='utf-8',buffering=1) as log:
        while tr.step<2000:
            row=tr.update((tr.step+1)%100==0);log.write(json.dumps(row)+'\n')
            if tr.step%100==0:print('DEEP_FP32',tr.step,round(row['loss'],4),round(row['seconds'],3),flush=True)
            if tr.step%1000==0:
                tr.save(OUT/f'checkpoint_{tr.step:06d}.pt');fp32_review(tr,data,f'fp32_{tr.step:06d}');gate.evaluate(tr,data,f'ptq_diagnostic_{tr.step:06d}')


if __name__=='__main__':main()
