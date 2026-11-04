"""Test a spatial decision in the existing360p DW, not an output filter.

Positive odd-sum integer kernels retain exact complementary decisions. Compare
straight-line phase error/width and actual source crops; smoothing alone does
not certify image quality. Uses the current ordinal-vote checkpoint unchanged
apart from this existing convolution's weights and bias.
"""
import argparse,copy
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .audit_b1_binary_boundary_phase import contour_metrics
from .b1_binary_layout import fields_for_layout
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m

KERNELS={
 'point7':[[0,0,0],[0,7,0],[0,0,0]],
 'cross7':[[0,1,0],[1,3,1],[0,1,0]],
 'box9':[[1,1,1],[1,1,1],[1,1,1]],
 'isotropic13':[[1,1,1],[1,5,1],[1,1,1]],
}

@torch.no_grad()
def install(net,kernel):
    k=torch.as_tensor(kernel,device=net.weight['up2_low_dw'].device);s=int(k.sum())
    assert s%2==1 and k.min()>=0
    key='up2_low_dw';net.weight[key].zero_();net.bias[key].zero_();getattr(net,key+'_m').fill_(131071)
    net.weight[key][:15,0]=k*131071/131072;net.bias[key][:15]=127*(1-s)//2*131071/131072
    dot=torch.arange(127*s+1);b=127*(1-s)//2
    ramp=lambda x:net.round((x.double()+b)*131071/131072).clamp(0,127)
    assert torch.equal(ramp(dot)+ramp(127*s-dot),torch.full_like(ramp(dot),127))

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    torch.set_num_threads(2);dest=Path(a.out);dest.mkdir(exist_ok=False);state,base=load(a.checkpoint);setup=state['binary_coverage720']
    nets={name:copy.deepcopy(base) for name in KERNELS}
    for name,net in nets.items():install(net,KERNELS[name])
    matrix=torch.tensor(setup['matrix']).float();bias=torch.tensor(setup['bias']).float()
    levels=torch.cartesian_prod(torch.arange(15),torch.arange(9),torch.arange(9)).float();palette=(levels@matrix.T+bias).clamp(-128,127)
    code=torch.arange(1215)[:,None,None];constant=F.pad(fields_for_layout(code,setup)*127,(0,0,0,0,0,1))
    for net in nets.values():assert torch.equal(fragment(net,constant,'up1_low_pw')['rgb_head'][:,:,0,0],palette)
    yy,xx=torch.meshgrid(torch.arange(96),torch.arange(128),indexing='ij');records=[]
    for name,net in nets.items():
        for pair,(ca,cb) in enumerate((((5,4,4),(9,4,4)),((5,3,4),(6,6,3)),((2,4,5),(11,4,3)))):
            c0=(matrix@torch.tensor(ca).float()+bias).clamp(-128,127);c1=(matrix@torch.tensor(cb).float()+bias).clamp(-128,127)
            for slope in (.2,.5,1.):
                for phase in range(4):
                    truth=64+slope*(torch.arange(96)-48)+phase-.5
                    code=torch.where(xx>truth[:,None],cb[0]*81+cb[1]*9+cb[2],ca[0]*81+ca[1]*9+ca[2])[None]
                    z=F.pad(F.avg_pool2d(fields_for_layout(code,setup)*127,4),(0,0,0,0,0,1));z=net.round(z)
                    rgb=fragment(net,z,'up1_low_pw')['rgb_head'][0]
                    records.append(dict(variant=name,pair=pair,slope=slope,phase=phase,**contour_metrics(rgb,c0,c1,truth)))
        rows=[r for r in records if r['variant']==name];print('JOINT_SPATIAL_RAMP',name,{k:float(np.mean([r[k] for r in rows])) for k in ('contour_rms_px','boundary_width_px','backtrack')},flush=True)
    data={r['id']:r for r in m.load_data(a.data)['validation']};board=Image.new('RGB',(1680,1256),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13)
    for j,(name,net) in enumerate(nets.items()):
        raw=net.export();actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cpu');assert len(raw)==20480
        outputs={}
        for rid in set(c[0] for c in m.t.r.c.CROPS):
            x=torch.from_numpy(m.t.r.read(data[rid]['path']).transpose(2,0,1).copy())[None].float()-128
            nodes=net(x/128,all_nodes=True);other,_=actual(x,True);assert all(torch.equal(v,other[k]) for k,v in nodes.items())
            outputs[rid]=m.t.r.c.rgb_image(nodes['rgb_head'][0]/128);outputs[rid].save(dest/(name+'_'+rid+'.png'))
        for i,(rid,box,label) in enumerate(m.t.r.c.CROPS):
            board.paste(outputs[rid].crop(box),(j*420,i*314+34));draw.text((j*420+3,i*314+4),label+' | '+name,font=font,fill='white')
        saved=copy.deepcopy(state);saved.update(model=net.state_dict(),joint_spatial_ramp_kernel=KERNELS[name],visual_acceptance=False)
        torch.save(saved,dest/(name+'.pt'))
    board.save(dest/'source_crops.png');write_json(dest/'review.json',dict(records=records,all_constant1215_error=0,
      actual_source_node_error=0,byte_size=20480,complement_error=0,extra_inference_ops=0,
      checkpoint_sha256=file_hash(a.checkpoint),visual_acceptance=False))

if __name__=='__main__':main()
