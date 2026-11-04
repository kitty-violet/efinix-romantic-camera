"""Check a deliberately cool, contrasting colour code (targets only)."""
import argparse
import shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_categorical_palette import nearest
from . import train_b1_minimum_area as m
from .runtime import configure_workspace,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);a=ap.parse_args()
    configure_workspace();dest=m.OUT/a.name;dest.mkdir(exist_ok=False);data=m.load_data()
    base=np.array([25.,24.,38.])
    endpoints=np.array([[244,237,225],[155,181,213],[125,140,159],[180,220,85],
                        [73,126,71],[201,176,144],[170,120,113],[89,130,197]],np.float64)
    candidates={};images={r['id']:[Image.fromarray(m.t.r.read(r['target']))] for r in data['validation']}
    records=[]
    for levels in ((.625,1.),(.25,.5,.75,1.)):
        features=np.zeros((1+8*len(levels),9));features[:,-1]=1
        for ray in range(8):features[1+ray*len(levels):1+(ray+1)*len(levels),ray]=levels
        weights=np.vstack((endpoints-base,base));palette=(features@weights).clip(0,255)
        candidates[str(len(palette))]=dict(levels=list(levels),features=features.tolist(),weights=weights.tolist(),palette=palette.tolist())
        for row in data['validation']:
            target=m.t.r.read(row['target']);out=palette[nearest(target,palette)].reshape(target.shape).round().clip(0,255).astype(np.uint8)
            im=Image.fromarray(out);im.save(dest/(row['id']+f'_{len(palette)}_target_only.png'));images[row['id']].append(im)
            records.append(dict(id=row['id'],colours=len(palette),mae=float(np.abs(out.astype(np.float64)-target).mean())))
    board=Image.new('RGB',(1260,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        for j,title in enumerate(('离线目标','自设17色·容量诊断','自设33色·容量诊断')):
            board.paste(images[key][j].crop(box),(420*j,314*i+34));draw.text((420*j+4,314*i+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'target-capacity.png');shutil.copy2(__file__,dest/Path(__file__).name)
    write_json(dest/'results.json',dict(candidates=candidates,records=records,target_only=True,source_prediction=False,
        palette_hand_designed=True,network_trained=False,hardware_inference_verified=False,visual_acceptance=False))
    print('DESIGNED_PALETTE_CAPACITY',records,flush=True)


if __name__=='__main__':main()
