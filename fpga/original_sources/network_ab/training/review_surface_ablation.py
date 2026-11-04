"""Compare only the same two frozen validation frames across trained branches."""
import argparse
import json
from pathlib import Path
from PIL import Image,ImageDraw
from .runtime import file_hash,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    modes=('light','texture','combined');old=Path(r'E:\fpga\model-training-workspace\runs\coherent-texture-20261002\guarded_train\validation_000250')
    logs={mode:[json.loads(x) for x in (a.root/f'{mode}_train/metrics_from_000000.jsonl').read_text().splitlines()][:200] for mode in modes}
    if any(len(v)!=200 for v in logs.values()):raise ValueError('All branches must complete 200 steps')
    if not all([r['sample'] for r in logs['light']]==[r['sample'] for r in logs[m]] for m in modes):
        raise ValueError('Ablation sample sequence differs')
    out=a.root/'ablation_review';out.mkdir(exist_ok=True)
    records=[]
    for filename in ('VID_20260925_095525.jpg','VID_20260925_101046.jpg'):
        entries=[('PREVIOUS 4250',old/filename)]+[(m.upper()+' 200',a.root/f'{m}_train/validation_000200'/filename) for m in modes]
        board=Image.new('RGB',(1280,4*380),'#151515');draw=ImageDraw.Draw(board)
        for i,(label,path) in enumerate(entries):
            im=Image.open(path).crop((1280,0,2560,720))
            board.paste(im.resize((640,360),Image.Resampling.LANCZOS),(0,i*380+20))
            board.paste(im.crop((320,280,960,640)),(640,i*380+20))
            draw.text((4,i*380+3),label+' / FULL VIEW LEFT, NATIVE CENTER CROP RIGHT',fill='white')
        dst=out/filename;board.save(dst,quality=97);records.append({'path':str(dst),'sha256':file_hash(dst)})
    write_json(out/'review.json',{'status':'VISUAL_SELECTION_PENDING','same_200_training_samples':True,
      'modes':list(modes),'records':records,'notes':'Validation-only ablation; no test-scene outputs used for selection.'})
    print(json.dumps({'same_samples':True,'output':str(out)}),flush=True)


if __name__=='__main__':main()
