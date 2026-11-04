"""Reduce only low-pass spatial mixing inside B1, preserving each filter's DC."""
import copy
from types import SimpleNamespace
import torch
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_19_levels as n
from . import train_b1_learned_colour as c
from .runtime import configure_workspace,write_json,file_hash

OUT=n.OUT/'internal-blur-probes'


@torch.no_grad()
def modify(net,keys,alpha):
    records=[]
    for key in keys:
        w=net.weight[key];dc=w.flatten(1).sum(1);neg=(-w.clamp_max(0)).flatten(1).sum(1)
        selected=(dc>1e-5)&(neg<.15*dc)
        old=w.clone();centre=torch.zeros_like(w);centre[:,:,1,1]=dc[:,None]
        w[selected]=alpha*w[selected]+(1-alpha)*centre[selected]
        m=(w.flatten(1).abs().amax(1)*131072/112).ceil().clamp(1,131071)
        current=getattr(net,key+'_m');current[:,selected]=m[selected].double()[None,:,None,None]
        assert torch.allclose(w.flatten(1).sum(1),old.flatten(1).sum(1),atol=1e-6)
        records.append(dict(layer=key,channels=selected.nonzero().flatten().tolist(),alpha=alpha))
    return records


@torch.inference_mode()
def main():
    configure_workspace();OUT.mkdir(exist_ok=False);data,selected,cache=c.load_data()
    src=n.OUT/'chroma-calibration/package/coefficients-padded.bin';original=c.ColourB1(False,src).cuda().eval();original.quantized=True
    enc=('down1__dw','down2__dw');dec=('up1_high_dw','up2_low_dw','up2_high_dw')
    variants=[('calibrated',(),1),('decoder070',dec,.7),('decoder040',dec,.4),('encoder070',enc,.7),('both070',enc+dec,.7)]
    board=Image.new('RGB',(2100,1248),'#17202a');d=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16);records=[]
    for col,(name,keys,alpha) in enumerate(variants):
        net=copy.deepcopy(original);changes=modify(net,keys,alpha);tr=SimpleNamespace(net=net,cached=cache,step=13700)
        dest=OUT/name;c.review(tr,data,selected,dest,integer=True,oracle_comparison=False)
        img=Image.open(dest/'reviewed.png')
        for row in range(4):
            board.paste(img.crop((840,row*312+32,1260,row*312+312)),(420*col,row*312+32));d.text((420*col+3,row*312+4),name,font=font,fill='white')
        records.append(dict(name=name,changes=changes,sha256=file_hash(dest/'coefficients-padded.bin'),trained_after_change=False,visual_accepted=False))
        del tr,net;torch.cuda.empty_cache()
    board.save(OUT/'probes.png');write_json(OUT/'binding.json',dict(parent=str(src),variants=records))
    print('INTERNAL_BLUR_PROBES_COMPLETE',flush=True)


if __name__=='__main__':main()
