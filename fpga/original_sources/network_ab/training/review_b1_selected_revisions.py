"""Review the selected actual byte package on all reserved-source stills.

These sources were used in historical reconstruction work, hence NOT blind.
This review does not choose a fourth design or change any coefficients.
"""
import argparse,shutil
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw,ImageFont
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    for k in ('package','data','out'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();torch.set_num_threads(2);out=Path(a.out);out.mkdir(exist_ok=False);data=m.load_data(a.data);raw=Path(a.package).read_bytes();net=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu');records=[]
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    for row in data['reserved_test']:
        assert file_hash(row['path'])==row['sha256'] and file_hash(row['target'])==row['target_sha256']
        x=m.t.r.read(row['path']);t=m.t.r.read(row['target']);p=net(torch.tensor(x.transpose(2,0,1)[None].copy()).float()-128)
        y=(p[0].permute(1,2,0).numpy()+128).astype(np.uint8);im=Image.fromarray(y);im.save(out/(row['id']+'_output.png'))
        board=Image.new('RGB',(1920,390),'#17202a');draw=ImageDraw.Draw(board)
        for j,(label,z) in enumerate((('原片',Image.fromarray(x)),('目标',Image.fromarray(t)),('A · 实际B1',im))):board.paste(z.resize((640,360),Image.Resampling.LANCZOS),(640*j,30));draw.text((640*j+4,4),label+' | '+row['id'],font=font,fill='white')
        board.save(out/(row['id']+'_comparison.png'));records.append(dict(id=row['id'],source=row['path'],target=row['target'],mae=float(np.abs(y.astype(float)-t).mean()),source_sha256=row['sha256'],target_sha256=row['target_sha256']))
        print('RESERVED_ACTUAL_A',row['id'],flush=True)
    shutil.copy2(__file__,out/Path(__file__).name);write_json(out/'review.json',dict(records=records,package_sha256=file_hash(a.package),same_selected_coefficients=True,no_refit=True,not_blind=True,historical_reconstruction_usage=True))

if __name__=='__main__':main()
