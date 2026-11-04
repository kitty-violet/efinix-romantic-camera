"""Compare encoded B1 edge transition width across source sampling phases."""
import argparse
import torch
from . import train_b1_minimum_area as m
from .runtime import configure_workspace,write_json,file_hash


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoints',nargs='+',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    configure_workspace();torch.set_num_threads(2);results=[]
    for path in a.checkpoints:
        state=torch.load(path,map_location='cpu',weights_only=True);q=m.t.r.initialize();q.load_state_dict(state['model']);q.quantized=True
        q.package['residual_encoding']=state['residual_encoding'];net=m.t.r.IntegerB1(m.t.r.runtime_from_image(q.export()));rows=[]
        for phase in range(4):
            x=torch.full((1,3,64,128),-64.,device='cuda');x[:,:,:,60+phase:]=64
            out=net(x)[0].mean(0)[32];low=out[16:32].mean();high=out[96:112].mean();p=(out-low)/(high-low)
            transition=int(((p>.1)&(p<.9)).sum());edge=int((p>=.5).nonzero()[0,0])
            rows.append(dict(phase=phase,width10_to90_pixels720=transition,edge_position=edge,input_edge=60+phase,
                colour_contrast=float(high-low)))
        results.append(dict(checkpoint=path,sha256=file_hash(path),actual_bytes=True,rows=rows))
        print('INTEGER_EDGE_TRANSITION',path,rows,flush=True)
    write_json(a.output,dict(results=results,scope='Synthetic edge regression; not a substitute for scene/video review'))


if __name__=='__main__':main()
