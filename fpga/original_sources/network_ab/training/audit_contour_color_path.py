"""Causal tests of the actual frozen renderer, not a diagram interpretation."""
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from torch.nn import functional as F
from .stroke_color_input_net import StrokeColorInputNet
from .train_native_contours import ROOT,tensor,load_rgb
from .train_b1_distillation import to_rgb
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible


@torch.no_grad()
def main():
    configure_workspace();contract_guard();set_reproducible(2026100487);torch.set_num_threads(4);cv2.setNumThreads(2)
    source=ROOT/'native-stroke-color-texture-v4';out=ROOT/'contour-color-path-audit-v1';out.mkdir(exist_ok=False)
    state=torch.load(source/'generator_fp32.pt',map_location='cpu',weights_only=True)
    assert all(file_hash(p)==h for p,h in state['binding']['code'].items())
    net=StrokeColorInputNet(False).cuda().eval();net.load_state_dict(state['model'])
    data=json.loads((source/'data.json').read_text());records=[]
    for row in data['validation']:
        x=tensor(load_rgb(row['source']))[None].cuda();normal=net(x);expected=load_rgb(source/'eval-001600'/(row['id']+'.png'))
        assert np.array_equal(to_rgb(normal[0]),expected)
        variants={'normal':normal,'zero_contour_cue':net(x,ablate_ink=True)}
        hook=net.ink_features.register_forward_pre_hook(lambda m,a:(torch.roll(a[0],(47,83),(-2,-1)),))
        variants['displaced_contour_cue']=net(x);hook.remove()
        hook=net.color_input.register_forward_hook(lambda m,a,y:torch.zeros_like(y))
        variants['remove_RGB_entrance']=net(x);hook.remove()
        hook=net.color_input.register_forward_pre_hook(lambda m,a:(F.avg_pool2d(F.pad(a[0],(4,4,4,4),mode='replicate'),9,1),))
        variants['remove_RGB_entrance_fine_detail']=net(x);hook.remove()
        rec=dict(id=row['id'],reload_uint8_max_error=0,variants={})
        for k,y in variants.items():
            Image.fromarray(to_rgb(y[0])).save(out/(row['id']+'_'+k+'.png'))
            rec['variants'][k]=dict(rgb255_mae_from_normal=float((y-normal).abs().mean()*128),
                maximum_rgb255_difference=float((y-normal).abs().max()*128))
        records.append(rec)
    f=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16);b=Image.new('RGB',(1600,948),'#17202a');d=ImageDraw.Draw(b)
    cases=[('val_stairs_090',(140,380,460,660)),('val_stairs_090',(420,180,740,460)),('val_plaza_090',(80,400,400,680))]
    names=['normal','zero_contour_cue','displaced_contour_cue','remove_RGB_entrance','remove_RGB_entrance_fine_detail']
    titles=['旧模型正常输出','轮廓输入置零','轮廓错移83×47像素','移除原片颜色入口','仅去掉入口细碎纹理']
    for j,(key,box) in enumerate(cases):
        for i,(name,title) in enumerate(zip(names,titles)):
            d.text((320*i+4,316*j+4),title,font=f,fill='white');b.paste(Image.open(out/(key+'_'+name+'.png')).crop(box),(320*i,316*j+34))
    b.save(out/'causal_comparison.png')
    write_json(out/'report.json',dict(records=records,weight_path=str(source/'generator_fp32.pt'),weight_sha256=file_hash(source/'generator_fp32.pt'),
        code=str(Path(__file__)),code_sha256=file_hash(__file__),
        actual_path='features = ReLU(native_features + quarter_color + ink_features(logits/8) + color_input(original_RGB)); renderer -> RGB',
        derived_contour_figure='Canny(fully generated RGB), diagnostic only; was never an input to this existing renderer',
        classification_probability='sigmoid(logit), confidence rather than physical black-ink intensity',
        no_deployment_claim=True))
    print('COLOR_PATH_AUDIT_DONE',json.dumps(records),flush=True)


if __name__=='__main__':main()
