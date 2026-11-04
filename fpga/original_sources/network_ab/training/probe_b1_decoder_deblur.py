"""Check stronger internal deblurring without altering graph or colour levels."""
import copy
from types import SimpleNamespace
import torch
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_19_crisp as crisp
from . import train_b1_learned_colour as c
from .probe_b1_internal_blur import modify
from .runtime import configure_workspace,write_json,file_hash


@torch.inference_mode()
def main():
    configure_workspace();out=crisp.OUT/'decoder-strength-probe';out.mkdir(exist_ok=False)
    data,selected,cache=c.load_data();src=crisp.OUT/'qat/package/coefficients-padded.bin'
    original=c.ColourB1(False,src).cuda().eval();original.quantized=True
    variants=[('trained',1.),('less_mixing',.2),('internal_unsharp',-.2)]
    board=Image.new('RGB',(1260,1248),'#17202a');d=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17);records=[]
    for col,(name,alpha) in enumerate(variants):
        net=copy.deepcopy(original)
        changes=modify(net,('up1_high_dw','up2_low_dw') if alpha!=1 else (),alpha)
        tr=SimpleNamespace(net=net,cached=cache,step=15700);dest=out/name
        c.review(tr,data,selected,dest,integer=True,oracle_comparison=False)
        img=Image.open(dest/'reviewed.png')
        for row in range(4):
            board.paste(img.crop((840,row*312+32,1260,row*312+312)),(420*col,row*312+32))
            d.text((420*col+4,row*312+4),name,font=font,fill='white')
        records.append(dict(name=name,changes=changes,sha256=file_hash(dest/'coefficients-padded.bin'),trained_after_change=alpha==1,visual_accepted=False))
        del tr,net;torch.cuda.empty_cache()
    board.save(out/'probes.png');write_json(out/'binding.json',dict(parent=str(src),variants=records))
    print('DECODER_STRENGTH_PROBE_COMPLETE',flush=True)


if __name__=='__main__':main()
