"""Separate contour causality from the information in oracle region colors."""
import argparse,json
from pathlib import Path
import torch
from PIL import Image,ImageDraw,ImageFont
from .contour_region_net import ContourRegionNet,scaffold_target
from .train_b1_distillation import read_image,as_tensor,to_rgb
from .audit_capacity_texture import texture_scores
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=ap.parse_args()
    configure_workspace();contract_guard();set_reproducible(2026100471)
    s=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    assert s['config']['graph_schema']=='contour-region-bottleneck-v1' and all(file_hash(k)==v for k,v in s['binding']['code'].items())
    target_fn=scaffold_target
    if s['config'].get('scaffold_schema')=='coarse-color8-and-thin-boundary-v2':
        from .train_sharp_contour_regions import scaffold_target as target_fn
    out=a.checkpoint.parent/f'audit-{s["step"]:06d}';out.mkdir(exist_ok=False)
    net=ContourRegionNet().to(a.device).eval();net.load_state_dict(s['model']);rows=[]
    data=json.loads((a.checkpoint.parent/'data.json').read_text(encoding='utf-8'))
    for r in data['validation']:
        x=as_tensor(read_image(r['source']))[None].to(a.device);t=as_tensor(read_image(r['target']))[None].to(a.device);g=net.encode(x);true=target_fn(t)
        variants=dict(actual=g,teacher_edges=torch.cat([g[:,:3],true[:,3:]],1),teacher_colors=torch.cat([true[:,:3],g[:,3:]],1),zero_edges=torch.cat([g[:,:3],g[:,3:]*0],1),oracle=true)
        actual=net.render(g);scores={}
        for key,v in variants.items():
            y=net.render(v);Image.fromarray(to_rgb(y[0])).save(out/(r['id']+'_'+key+'.png'))
            scores[key]=dict(**texture_scores(y,t),change_from_actual_rgb255=float((y-actual).abs().mean()*128))
        rows.append(dict(id=r['id'],variants=scores))
    board=Image.new('RGB',(1600,894),'#18202a');d=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    for j,(key,(x,y)) in enumerate([('val_stairs_090',(150,430)),('val_stairs_090',(424,320)),('val_plaza_090',(80,400))]):
        for i,(variant,title) in enumerate([('actual','真实网络'),('teacher_edges','诊断：仅换教师轮廓'),('teacher_colors','诊断：仅换教师区域颜色'),('zero_edges','诊断：去掉预测轮廓'),('oracle','诊断：两者都换成教师')]):
            d.text((320*i+4,298*j+5),title,font=font,fill='white');board.paste(Image.open(out/(key+'_'+variant+'.png')).crop((x,y,x+320,y+264)),(320*i,298*j+30))
    board.save(out/'comparison.png');write_json(out/'report.json',dict(checkpoint=str(a.checkpoint),checkpoint_sha256=file_hash(a.checkpoint),records=rows,interventions_are_not_inference=True,oracle_color_contains_teacher_simplification=True,scaffold_schema=s['config'].get('scaffold_schema','dense360-color-v1'),board_validated=False))
    print('CONTOUR_AUDIT_DONE',flush=True)


if __name__=='__main__':main()
