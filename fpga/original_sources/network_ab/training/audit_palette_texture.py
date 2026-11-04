"""Separate generator texture errors from fixed-palette amplification."""
import json
import torch
from PIL import Image,ImageDraw,ImageFont
from .clean_skip_capacity_net import CleanSkipNet
from .component_region_net import decode
from .train_luma_regions import ROOT,RUN,components,reconstruct
from .train_b1_distillation import read_image,as_tensor,to_rgb
from .audit_capacity_texture import texture_scores
from .runtime import file_hash,write_json


@torch.no_grad()
def main():
    torch.set_num_threads(2);base=ROOT/'runs'/RUN;ck=base/'teacher-clean-skip-v1/fp32_checkpoint_000500.pt';out=base/'teacher-palette-audit-v1';out.mkdir(exist_ok=False)
    s=torch.load(ck,map_location='cpu',weights_only=True);assert all(file_hash(k)==v for k,v in s['binding']['code'].items());net=CleanSkipNet().eval();net.load_state_dict(s['model']);data=json.loads((ck.parent/'data.json').read_text(encoding='utf-8'));rows=[]
    for r in data['validation']:
        x=as_tensor(read_image(r['source']))[None];t=as_tensor(read_image(r['target']))[None];p=net(x)
        cases={'teacher':t,'teacher_palette':decode(components(t)),'student_raw':reconstruct(*p.unbind(1)),'student_palette':decode(p)}
        for tag,y in cases.items():
            Image.fromarray(to_rgb(y[0])).save(out/(r['id']+'_'+tag+'.png'));rows.append(dict(id=r['id'],case=tag,**texture_scores(y,t)))
        print('PALETTE_AUDIT',r['id'],flush=True)
    board=Image.new('RGB',(1280,1192),'#18202a');d=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    for j,(key,(x,y)) in enumerate([('val_stairs_090',(150,430)),('val_stairs_090',(424,320)),('val_plaza_090',(80,400)),('val_new_hallway_120',(700,280))]):
        for i,(tag,title) in enumerate([('teacher','原教师'),('teacher_palette','教师直接过当前色阶'),('student_raw','学生：色阶之前'),('student_palette','学生：色阶之后')]):
            d.text((i*320+5,j*298+5),title,font=font,fill='white');board.paste(Image.open(out/(key+'_'+tag+'.png')).crop((x,y,x+320,y+264)),(i*320,j*298+30))
    board.save(out/'comparison.png');write_json(out/'report.json',dict(records=rows,checkpoint_sha256=file_hash(ck),script_sha256=file_hash(__file__),reference_is_continuous_original_teacher=True,raw_output_is_diagnostic_not_deployment=True,not_blind=True))


if __name__=='__main__':main()
