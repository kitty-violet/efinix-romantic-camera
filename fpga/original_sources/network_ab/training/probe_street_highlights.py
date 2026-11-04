"""Same decoded source frame, fixed HDR exposure probes; production is untouched."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .media import decode_frames
from .runtime import file_hash,write_json


def main():
    root=Path(r'E:\fpga\model-training-workspace\runs\coherent-surface-diagnosis-20261002')
    render=json.loads(Path(r'E:\fpga\model-training-workspace\runs\coherent-texture-20261002\final-guarded\render.json').read_text())
    row=next(r for r in render['clips'] if r['id']=='test_street')
    if file_hash(row['source'])!=row['source_sha256']:raise ValueError('Source drift')
    frame=row['source_frame_indices'][0]+75;records=[]
    board=Image.new('RGB',(1280,750*3),'#151515');draw=ImageDraw.Draw(board)
    for i,ev in enumerate((-1.,-2.,-3.)):
        decoded=list(decode_frames(row['source'],indices=[frame],hlg_exposure_stops=ev))
        if len(decoded)!=1 or decoded[0][0]!=frame:raise ValueError('Frame mismatch')
        index,pts,pixels=decoded[0]
        im=Image.fromarray(pixels).resize((1280,720),Image.Resampling.BICUBIC)
        image=np.array(im).astype(np.float32);sky=image[20:140,400:500]
        lum=sky@np.array([.2126,.7152,.0722],np.float32)
        path=root/f'input_only_exposure_{int(ev)}.png';im.save(path)
        records.append({'ev':ev,'index':index,'pts':pts,'path':str(path),
          'sky_luma_mean':float(lum.mean()),'sky_luma_std':float(lum.std()),
          'sky_min':float(lum.min()),'sky_max':float(lum.max())})
        board.paste(im,(0,750*i+25));draw.text((5,750*i+5),f'INPUT ONLY / fixed exposure {ev:g} EV / same source frame {frame}',fill='white')
    board.save(root/'input_exposure_probe.jpg',quality=95)
    write_json(root/'input_exposure_probe.json',{'source':row['source'],
      'source_sha256':row['source_sha256'],'records':records,
      'note':'Diagnostic only; no approved dataset/preprocessing changes. Does not prove true scene detail recovery.'})
    print(json.dumps(records),flush=True)


if __name__=='__main__':main()
