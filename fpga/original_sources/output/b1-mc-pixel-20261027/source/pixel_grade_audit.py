"""Read-only grading reference for the exact user-approved pixel coefficients."""
from pathlib import Path
import json,csv,hashlib
from PIL import Image,ImageDraw,ImageFont
import numpy as np

OUT=Path(__file__).resolve().parents[1]

def main():
    d=OUT/'pixel-grading-reference';d.mkdir(exist_ok=True)
    source=OUT/'approved-pixel-review/colour_controls.json'
    curve=json.loads(source.read_text('utf-8'))['flat_r3']['gray_curve_mean'];groups=[];start=0;prev=curve[0]
    for i in range(1,256):
        if abs(curve[i]-prev)>.1:groups.append(dict(input_start=start,input_end=i-1,output_gray=prev));start=i;prev=curve[i]
    groups.append(dict(input_start=start,input_end=255,output_gray=prev))
    with (d/'current-gray-curve.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['input_gray','actual_B1_mean_RGB']);w.writerows(enumerate(curve))
    raw=OUT/'exports/flat-r3-user-selected/coefficients-padded.bin';digest=hashlib.sha256(raw.read_bytes()).hexdigest()
    status=dict(coefficient_sha256=digest,source_measurement=str(source),gray_groups=groups,
        grading_reference=dict(contrast_relative=1.,saturation_relative=1.,brightness_offset=0,definition='relative to the currently approved coefficients, not pre-existing independent model sliders'),
        saturation_model='shared YCoCg palette and linear chroma readout; relative S multiplies that current chroma contribution',
        high_priority='compress bright bins, modest contrast and saturation increase, maintain crisp separated colour masses',
        high_bins_current=[dict(input=[g['input_start'],g['input_end']],output=g['output_gray']) for g in groups[-3:]],
        applied_changes=False)
    (d/'reference.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    im=Image.new('RGB',(1320,860),'#f8faff');draw=ImageDraw.Draw(im)
    font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
    draw.text((60,22),'纯像素 R3：当前实际 B1 亮度曲线',font=font(30),fill='#20354e')
    draw.text((60,68),'对比度基准 100%   ·   饱和度基准 100%   ·   亮度偏移 0（均相对当前认可版）',font=font(20),fill='#52647d')
    left,top,width,height=95,145,790,580
    xy=lambda x,y:(round(left+x/255*width),round(top+(255-y)/255*height))
    draw.rectangle((left,top,left+width,top+height),fill='white',outline='#9aaec7')
    for v in [0,32,64,96,128,160,192,224,255]:
        x,y=xy(v,v);draw.line((x,top,x,top+height),fill='#e0e7ef');draw.line((left,y,left+width,y),fill='#e0e7ef')
        draw.text((x-13,top+height+12),str(v),font=font(17),fill='#303947');draw.text((left-45,y-10),str(v),font=font(17),fill='#303947')
    for v in range(0,255,8):draw.line((xy(v,v),xy(min(v+4,255),min(v+4,255))),fill='#8997a7',width=3)
    draw.line([xy(i,v) for i,v in enumerate(curve)],fill='#176fd0',width=4)
    draw.text((370,785),'输入灰度（0黑 → 255白）',font=font(20),fill='#303947')
    draw.text((95,112),'输出灰度',font=font(19),fill='#303947')
    draw.text((940,145),'当前均匀灰块实测',font=font(24),fill='#20354e')
    for j,g in enumerate(groups):
        draw.text((940,195+49*j),f"{g['input_start']:3d}–{g['input_end']:3d}  →  {g['output_gray']:.1f}",font=font(21),fill='#176fd0')
    draw.text((940,625),'优先调整：',font=font(22),fill='#20354e')
    draw.multiline_text((940,665),'压低亮灰与白色两档\n适度加对比、加饱和\n保持色块边界清楚\n本次仅测量，尚未调色',font=font(20),fill='#52647d',spacing=13)
    draw.text((95,831),'蓝线 = 实际整数B1；灰虚线 = 输入不变。普通画面还受邻域归并影响。',font=font(16),fill='#64758e')
    im.save(d/'current-gray-curve.png')
    print('PIXEL_GRADING_REFERENCE',json.dumps(status,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
