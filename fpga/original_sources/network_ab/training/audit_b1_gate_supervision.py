"""Compare the signal actually supervised at B1's native gate grid."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from .train_b1_deep_contour import DeepStudent,OUT
from .train_b1_structural_gate import CROPS
from .runtime import configure_workspace,write_json,file_hash
from .engine import set_reproducible


@torch.no_grad()
def main():
    configure_workspace();set_reproducible(42)
    data=json.loads((OUT/'data.json').read_text(encoding='utf-8'));net=DeepStudent().cuda().eval()
    checkpoint=OUT/'checkpoint_002000.pt';net.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model'])
    dest=OUT/'supervision-grid-audit';dest.mkdir(exist_ok=False);records=[]
    for r in data:
        if r['split']=='train':continue
        rgb=np.array(Image.open(r['source']));x=torch.tensor(rgb.transpose(2,0,1)[None].copy(),device='cuda',dtype=torch.float32)/128-1
        # These constants match the actual optimized FeatureTraining objective.
        target=np.maximum(np.load(r['label']).astype(float)[::4,::4]*32-8,0)
        fields={'target':target}
        for name,quantized in [('fp32',False),('int8',True)]:
            net.quantized=quantized;fields[name]=net(x).clamp_min(0)[0,0].cpu().numpy()
        row=dict(id=r['id'],metrics={})
        for name,a in fields.items():
            Image.fromarray(np.clip(a*255/24,0,255).astype(np.uint8)).resize((1280,720),Image.Resampling.NEAREST).save(dest/(r['id']+'_'+name+'.png'))
            if name=='target':continue
            p=a>4;t=target>4
            row['metrics'][name]=dict(precision=float((p&t).sum()/max(p.sum(),1)),recall=float((p&t).sum()/max(t.sum(),1)),
                false_active_on_zero_target=float(((a>0)&(target==0)).sum()/max((target==0).sum(),1)),mae=float(abs(a-target).mean()))
        records.append(row)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18);board=Image.new('RGB',(1260,1256),'#17202a');draw=ImageDraw.Draw(board)
    for i,(key,box,label) in enumerate(CROPS):
        for j,(name,title) in enumerate([('target','同网格实际监督'),('fp32','FP32实际预测'),('int8','INT8实际预测')]):
            board.paste(Image.open(dest/(key+'_'+name+'.png')).convert('RGB').crop(box),(420*j,314*i+34));draw.text((420*j+5,314*i+4),label+' '+title,font=font,fill='white')
    board.save(dest/'native_same_grid.png')
    write_json(dest/'audit.json',dict(checkpoint_sha256=file_hash(checkpoint),code_sha256=file_hash(__file__),
        metric_correction='Use desired gate >4 on both sides; old label>0.12 vs prediction>4 was not a matched threshold',
        actual_supervision='max(32*soft_distance_label[::4,::4]-8,0)',visual_gate_gain=255/24,records=records))
    print('SUPERVISION_GRID_AUDIT_COMPLETE',flush=True)


if __name__=='__main__':main()
