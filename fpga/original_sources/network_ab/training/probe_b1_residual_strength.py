"""Causal coefficient probes of the old rank-initialized residual blocks.

These are explicitly untrained interventions, not accepted checkpoints.
Each probe still exports and executes only the actual B1 coefficient image.
"""
import copy
from types import SimpleNamespace
from PIL import Image,ImageDraw,ImageFont
import torch
from . import train_b1_sampling_colour_v2 as base
from . import train_b1_lattice_recovery as recovery
from . import train_b1_learned_colour as c
from .runtime import configure_workspace,write_json,file_hash


@torch.inference_mode()
def main():
    configure_workspace();out=base.OUT/'residual-probes';out.mkdir(exist_ok=False)
    data,selected,cache=c.load_data();source=recovery.OUT/'qat/package/coefficients-padded.bin'
    original=c.ColourB1(False,source).cuda().eval();original.quantized=True
    variants=[('trained',(1,1,1)),('all075',(.75,.75,.75)),('all050',(.5,.5,.5)),('last050',(1,1,.5)),('early050',(.5,.5,1))]
    board=Image.new('RGB',(420*5,312*4),'#17202a');d=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    records=[]
    for col,(name,scales) in enumerate(variants):
        net=copy.deepcopy(original)
        for i,scale in enumerate(scales,1):
            net.weight[f'res{i}_conv2__pw']*=scale;net.bias[f'res{i}_conv2__pw']*=scale
        tr=SimpleNamespace(net=net,cached=cache,step=10500)
        dest=out/name;c.review(tr,data,selected,dest,integer=True,oracle_comparison=False)
        package=Image.open(dest/'reviewed.png')
        for row in range(4):
            board.paste(package.crop((840,row*312+32,1260,row*312+312)),(col*420,row*312+32))
            d.text((col*420+4,row*312+4),name+' / '+str(scales),font=font,fill='white')
        records.append(dict(name=name,residual_factors=scales,package_sha256=file_hash(dest/'coefficients-padded.bin'),
                            trained_after_intervention=False,visual_accepted=False))
        del tr,net;torch.cuda.empty_cache()
    board.save(out/'probes.png');write_json(out/'binding.json',dict(parent=str(source),variants=records))
    print('RESIDUAL_PROBES_COMPLETE',flush=True)


if __name__=='__main__':main()
