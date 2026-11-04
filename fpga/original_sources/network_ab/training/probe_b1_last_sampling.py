"""Isolate how much the final nearest/refinement kernel softens real output."""
import copy
from types import SimpleNamespace
import torch
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_sampling_colour_v2 as b
from . import train_b1_learned_colour as c
from .runtime import configure_workspace,write_json,file_hash


@torch.inference_mode()
def main():
    configure_workspace();data,selected,cache=c.load_data();out=b.OUT/'last-sampling-probes';out.mkdir(exist_ok=False)
    src=b.OUT/'lattice-recovery/qat/package/coefficients-padded.bin';original=c.ColourB1(False,src).cuda().eval();original.quantized=True
    variants=[('trained',None),('half_interpolation',.5),('nearest',0),('limited_unsharp',-.1)]
    board=Image.new('RGB',(1680,1248),'#17202a');d=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16);records=[]
    for col,(name,alpha) in enumerate(variants):
        net=copy.deepcopy(original)
        if alpha is not None:
            gain=net.weight['up2_high_dw'].new_tensor([2/127]*6+[.5,.5])[:,None,None,None]
            central=net.weight['up2_high_dw'].new_zeros((8,1,3,3));central[:,:,1,1]=1
            if alpha>=0:w=alpha*net.weight['up2_high_dw']+(1-alpha)*central*gain
            else:
                kernel=central.new_tensor([[0,-.025,0],[-.025,1.1,-.025],[0,-.025,0]])[None,None]
                w=kernel.repeat(8,1,1,1)*gain
            net.weight['up2_high_dw'].copy_(w)
            m=(w.flatten(1).abs().amax(1)*131072/112).ceil().clamp(1,131071)
            net.up2_high_dw_m.copy_(m.double()[None,:,None,None])
        tr=SimpleNamespace(net=net,cached=cache,step=10500);dest=out/name
        c.review(tr,data,selected,dest,integer=True,oracle_comparison=False)
        img=Image.open(dest/'reviewed.png')
        for row in range(4):
            board.paste(img.crop((840,row*312+32,1260,row*312+312)),(420*col,row*312+32))
            d.text((420*col+3,row*312+4),name,font=font,fill='white')
        records.append(dict(name=name,alpha=alpha,sha256=file_hash(dest/'coefficients-padded.bin'),trained_after_probe=False,visual_accepted=False))
    board.save(out/'probes.png');write_json(out/'binding.json',dict(source=str(src),variants=records))
    print('LAST_SAMPLING_PROBES_COMPLETE',flush=True)


if __name__=='__main__':main()
