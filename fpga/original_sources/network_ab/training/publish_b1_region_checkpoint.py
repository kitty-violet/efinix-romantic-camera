"""Freeze a region-training checkpoint as an independently checked B1 package."""
import argparse
import shutil
from pathlib import Path
import torch
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .publish_b1_style_recovery import export
from .runtime import configure_workspace,file_hash,write_json


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--reserved',action='store_true');ap.add_argument('--device',choices=('cpu','cuda'),default='cuda');a=ap.parse_args()
    configure_workspace();torch.set_num_threads(2)
    src=Path(a.checkpoint);state=torch.load(src,map_location='cpu',weights_only=True)
    assert state.get('qat') or state.get('args',{}).get('qat')
    assert state.get('residual_encoding'),'Never infer missing residual metadata'
    net=m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin').to(a.device)
    net.load_state_dict(state['model']);net.package['residual_encoding']=state['residual_encoding'];net.quantized=True
    dest=Path(a.out);dest.mkdir(exist_ok=False);data=m.load_data()
    export(net,data['validation'],dest/'package')
    tensor=lambda rgb:torch.tensor(rgb.transpose(2,0,1)[None].copy(),device=a.device,dtype=torch.float32)/128-1
    folder=dest/'preview';folder.mkdir();images={}
    for row in data['validation']:
        rgb=m.t.r.read(row['path']);im=m.t.r.c.rgb_image(net(tensor(rgb))[0][0])
        im.save(folder/(row['id']+'_output.png'))
        images[row['id']]=[Image.fromarray(rgb),Image.open(row['target']),im]
    board=Image.new('RGB',(1260,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        for j,(im,title) in enumerate(zip(images[key],('原图','区域目标','实际B1输出'))):
            board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+4,314*i+5),label+' | '+title,font=font,fill='white')
    board.save(folder/'comparison.png')
    reserved=[]
    if a.reserved:
        folder=dest/'reserved-preview';folder.mkdir()
        actual=m.t.r.IntegerB1(m.t.r.runtime_from_image((dest/'package/coefficients-padded.bin').read_bytes()),device=a.device)
        for row in data['reserved_test']:
            x=tensor(m.t.r.read(row['path']))
            prediction=actual(x*128)/128;ref=net(x)[0]
            error=float((prediction-ref).abs().max());assert error==0
            m.t.r.c.rgb_image(prediction[0]).save(folder/(row['id']+'_actual_B1.png'))
            Image.open(row['path']).save(folder/(row['id']+'_source.png'))
            Image.open(row['target']).save(folder/(row['id']+'_target.png'))
            reserved.append(dict(id=row['id'],export_output_error=error))
    shutil.copy2(src,dest/'trained-checkpoint.pt');shutil.copy2(__file__,dest/Path(__file__).name)
    write_json(dest/'provenance.json',dict(checkpoint=str(src),checkpoint_sha256=file_hash(src),step=state['step'],
        package_sha256=file_hash(dest/'package/coefficients-padded.bin'),training_args=state.get('args'),
        residual_encoding=state['residual_encoding'],reserved_records=reserved,independent_region_boundaries=state.get('independent_region_boundaries',False),
        balanced_region_boundaries=state.get('balanced_region_boundaries',False),
        exported_bytes=20480,added_operators=0,postfilter=False,actual_board=False,visual_acceptance=False,verification_device=a.device,
        reserved_sources_used_for_current_training=False,reserved_sources_historically_used_for_reconstruction=True))
    print('REGION_PACKAGE_COMPLETE',str(dest),flush=True)


if __name__=='__main__':main()
