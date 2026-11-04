"""Small, explicitly labelled evidence panel; no generated or retouched pixels."""
import json
from PIL import Image,ImageDraw,ImageFont
from .train_luma_regions import ROOT,RUN
from .runtime import write_json


def main():
    base=ROOT/'runs'/RUN;run=base/'sharp-contour-region-v2';audit=run/'audit-001200'
    data=json.loads((run/'data.json').read_text(encoding='utf-8'));r=next(r for r in data['validation'] if r['id']=='val_stairs_090')
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17);board=Image.new('RGB',(1280,620),'#17202b');draw=ImageDraw.Draw(board)
    paths=[(r['target'],'目标教师'),(audit/'val_stairs_090_actual.png','学生真实输出'),(audit/'val_stairs_090_teacher_edges.png','诊断：仅替换教师轮廓'),(audit/'val_stairs_090_zero_edges.png','诊断：去掉轮廓')]
    for row,(x,y) in enumerate([(150,430),(424,320)]):
        for col,(path,title) in enumerate(paths):
            draw.text((col*320+5,row*298+6),title,font=font,fill='white');board.paste(Image.open(path).crop((x,y,x+320,y+264)),(col*320,row*298+30))
    draw.text((6,598),'仅第二列是学生实际推理；其余替换实验用于定位问题，不能作为上板效果。',font=font,fill='#f1cc79')
    board.save(run/'diagnosis.png')
    write_json(run/'method_review.json',dict(user_observation='student boundaries are broader and less precise than teacher',
        confirmed='dense-color oracle was not evidence that contour was sufficient; coarse-color variant makes contour intervention affect grass grouping',
        unresolved=['learned contour localization and thinness','clean region grouping','real temporal stability','quantization and complete FPGA implementation'],
        sources=[dict(url='https://github.com/SystemErrorWang/White-box-Cartoonization/blob/master/train_code/train.py',finding='separate training representations, not proof of an explicit detect-then-simplify inference pipeline'),
                 dict(url='https://github.com/hellozhuo/pidinet',finding='specialized learned pixel-difference edge detector'),
                 dict(url='https://raw.githubusercontent.com/hellozhuo/pidinet/master/models/ops.py',finding='central and angular pixel differences can fold into3x3 kernels; radial version uses5x5'),
                 dict(url='https://raw.githubusercontent.com/hellozhuo/pidinet/master/models/pidinet.py',finding='depthwise difference blocks and multiscale supervision; full official model has other operations not adopted')],
        pidinet_implemented=False,pidinet_downloaded=False,teacher_edge_intervention_is_diagnostic_only=True,style_accepted=False))
    print('CONTOUR_DIAGNOSIS_PUBLISHED',flush=True)


if __name__=='__main__':main()
