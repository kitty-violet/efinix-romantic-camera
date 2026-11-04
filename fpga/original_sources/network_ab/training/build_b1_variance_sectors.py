"""Four-quadrant convex-variance selection compiled into unchanged B1.

Kuwahara's least-variable neighbourhood is approximated with a convex PWL
quadratic: f(y)=y/4 + sum(max(y-k,0)/2), k=32,64,96, y in[0,127].
Unlike the earlier Laplacian-cost experiment this compares E[f(Y)]-f(E[Y]).
It is luminance-only, quantized and fixed-quadrant; NOT anisotropic Kuwahara.
All decisions, means and palette conversion are actual B1 coefficients.
"""
import argparse
import copy
import shutil
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .build_b1_ordered_palette import install as install_palette,GROUPS
from .build_b1_median_cartoon import LEVELS
from .probe_b1_decoder_coverage import fragment
from .runtime import write_json,file_hash,contract_guard


@torch.no_grad()
def build(net,wide=True,texture_cost=False,spread=False):
    for k,w in net.weight.items():w.zero_();net.bias[k].zero_()
    install_palette(net,LEVELS)
    z=net.weight['stem'];luma=z.new_tensor([.25,.5,.25]);knots=[0,32,64,96]
    for c in range(3):z[c,c,1,1]=.5
    net.bias['stem'][:3]=64
    for j,knot in enumerate(knots):
        z[3+j,:,1,1]=luma*.5;net.bias['stem'][3+j]=64-knot
    if texture_cost:
        # The linear term of f cancels in E[f(Y)]-f(E[Y]). Reuse that
        # channel for both signs of local texture, because a PWL variance
        # alone is blind to variation contained within one linear segment.
        z[3:8].zero_();net.bias['stem'][3:8].zero_()
        for j,knot in enumerate(knots[1:]):
            z[3+j,:,1,1]=luma*.5;net.bias['stem'][3+j]=64-knot
        lap=z.new_tensor([[0,-1,0],[-1,4,-1],[0,-1,0]])/8
        z[6]=luma[:,None,None]*lap;z[7]=-z[6]
    gaussian=z.new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
    net.weight['down1__dw'][:8 if texture_cost else 7,0]=gaussian
    for s in range(4):
        for c in range(3):net.weight['down1__pw'][4*s+c,c,0,0]=1
        net.weight['down1__pw'][4*s+3,3:7,0,0]=z.new_tensor([.25,.5,.5,.5])
        if texture_cost:
            net.weight['down1__pw'][4*s+3].zero_()
            net.weight['down1__pw'][4*s+3,3:6,0,0]=.5
            net.weight['down1__pw'][4*s+3,6:8,0,0]=.25
        q=z.new_zeros(3,3);y,x=s//2,s%2;q[y:y+2,x:x+2]=.25
        if spread:
            yy=z.new_tensor([6,1,1]) if y==0 else z.new_tensor([1,1,6])
            xx=z.new_tensor([6,1,1]) if x==0 else z.new_tensor([1,1,6])
            q=yy[:,None]*xx[None]/64
        net.weight['down2__dw'][4*s:4*s+4,0]=q
        net.weight['res1_conv1__dw'][4*s:4*s+4,0]=q if wide else torch.eye(3,device=z.device)*0
        if not wide:net.weight['res1_conv1__dw'][4*s:4*s+4,0,1,1]=1
    for c in range(16):net.weight['down2__pw'][c,c,0,0]=1
    #32 values: four sectors *[mean RGB, mean f(Y), four f(mean Y) hinges].
    key='res1_conv1__pw'
    for s in range(4):
        for c in range(4):net.weight[key][8*s+c,4*s+c,0,0]=1
        for j,knot in enumerate(knots):
            net.weight[key][8*s+4+j,4*s:4*s+3,0,0]=luma
            net.bias[key][8*s+4+j]=-knot
    net.weight['res1_conv2__dw'][:,0,1,1]=1
    key='res1_conv2__pw'
    for s in range(4):
        for c in range(3):net.weight[key][4*s+c,8*s+c,0,0]=1
        net.weight[key][4*s+3,8*s+3,0,0]=4
        net.weight[key][4*s+3,8*s+4:8*s+8,0,0]=z.new_tensor([-1,-2,-2,-2])
        if texture_cost:net.weight[key][4*s+3,8*s+4,0,0]=0
    # The next DW clamps any small negative variance caused by integer rounding.
    net.weight['res2_conv1__dw'][:16,0,1,1]=1
    key='res2_conv1__pw'
    for s in range(4):
        for c in range(3):net.weight[key][s*3+c,s*4+c,0,0]=1
    for p,(a,b) in enumerate(((3,7),(11,15))):
        net.weight[key][12+p,b,0,0]=127;net.weight[key][12+p,a,0,0]=-127;net.bias[key][12+p]=127
        net.weight[key][14+2*p,b,0,0]=1;net.weight[key][14+2*p,a,0,0]=-1
        net.weight[key][15+2*p,b,0,0]=1
    net.weight['res2_conv2__dw'][:18,0,1,1]=1
    key='res2_conv2__pw'
    for p in range(2):
        for c in range(3):
            j=p*6+c
            net.weight[key][j,j,0,0]=1;net.weight[key][j,12+p,0,0]=1;net.bias[key][j]=-127
            net.weight[key][j+3,j+3,0,0]=1;net.weight[key][j+3,12+p,0,0]=-1
        net.weight[key][12+p,15+2*p,0,0]=1;net.weight[key][12+p,14+2*p,0,0]=-1
    net.weight['res3_conv1__dw'][:14,0,1,1]=1
    key='res3_conv1__pw'
    for p in range(2):
        for c in range(3):
            net.weight[key][3*p+c,6*p+c,0,0]=1;net.weight[key][3*p+c,6*p+c+3,0,0]=1
    net.weight[key][6,13,0,0]=127;net.weight[key][6,12,0,0]=-127;net.bias[key][6]=127
    net.weight['res3_conv2__dw'][:7,0,1,1]=1
    key='res3_conv2__pw'
    for c in range(3):
        net.weight[key][c,c,0,0]=1;net.weight[key][c,6,0,0]=1;net.bias[key][c]=-127
        net.weight[key][c+3,c+3,0,0]=1;net.weight[key][c+3,6,0,0]=-1
    net.weight['up1_low_dw'][:6,0,1,1]=1
    matrix=z.new_tensor([[.25,.5,.25],[.5,0,-.5],[-.25,.5,-.25]])
    key='up1_low_pw';net.weight[key].zero_();net.bias[key].zero_()
    for g,(start,end) in enumerate(GROUPS):
        v=np.array(LEVELS[g],float);cuts=(v[:-1]+v[1:])/2
        raw=(cuts+8)/1.06/2 if g==0 else (cuts+8)/1.20/2 if g==1 else cuts/1.15/2
        for j,cut in enumerate(raw):
            net.weight[key][start+j,:3,0,0]=matrix[g]
            net.weight[key][start+j,3:6,0,0]=matrix[g]
            net.bias[key][start+j]=64-cut
    for b in range(1,4):net.package['residual_encoding'][f'res{b}_add']=dict(main_scaled=False,multiplier=1,post_multiplier=131071)
    preserved={'up1_high_dw','up2_low_dw','up2_low_pw','up2_high_dw','rgb_head'}
    for k,w in net.weight.items():
        if k in preserved:continue
        maximum=w.abs().flatten(1).amax(1)
        scale=torch.pow(2,torch.ceil(torch.log2(maximum.clamp_min(1e-5)/127)))
        getattr(net,k+'_m').copy_((scale*131072).clamp(1,131071).double()[None,:,None,None])


@torch.inference_mode()
def verify(net):
    gen=torch.Generator(device='cpu').manual_seed(20261006331)
    z=torch.zeros(64,32,8,8);z[:,:16]=torch.randint(0,128,(64,16,8,8),generator=gen)
    z[0,3:16:4]=7;z[1,3:16:4]=0;z[2,3:16:4]=127
    nodes=fragment(net,z,'res1_conv2.pw');out=nodes['up1_low_dw']
    cost=z[:,3:16:4];winner=cost.argmin(1)
    colours=torch.stack([z[:,4*s:4*s+3] for s in range(4)],2)
    expected=colours.gather(2,winner[:,None,None].expand(-1,3,1,-1,-1)).squeeze(2)
    gap=float((out[:,:3]+out[:,3:6]-expected).abs().max());assert gap==0,gap
    return dict(selection_cases=4096,exact_integer_selection_error=gap,tie_break='first')


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    ap.add_argument('--texture-cost',action='store_true');ap.add_argument('--spread',action='store_true');a=ap.parse_args()
    contract_guard();torch.set_num_threads(2);dest=Path(a.out);dest.mkdir(exist_ok=False)
    rows=m.load_data()['validation'];images={};records=[]
    for wide in (False,True):
        net=m.t.r.c.ColourB1(revive=False,source=m.t.r.OUT/'initial_identity.bin');net.quantized=True
        build(net,wide,a.texture_cost,a.spread);check=verify(net);sub=dest/('wide' if wide else 'base');sub.mkdir()
        actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(net.export()),device='cpu')
        node_error=0
        for row in rows:
            rgb=m.t.r.read(row['path']);x=torch.tensor(rgb.transpose(2,0,1)[None].copy(),dtype=torch.float32)-128
            real,_=actual(x,True);ref=net(x/128,all_nodes=True)
            errors={k:float((real[k]-v).abs().max()) for k,v in ref.items() if k!='input'}
            node_error=max(node_error,max(errors.values()));assert node_error==0
            image=m.t.r.c.rgb_image(real['rgb_head'][0]/128);image.save(sub/(row['id']+'_actual_B1.png'))
            images[row['id'],wide]=image
        (sub/'coefficients-padded.bin').write_bytes(net.export())
        torch.save(dict(model=copy.deepcopy(net.state_dict()),qat=True,step=0,residual_encoding=net.package['residual_encoding'],
            ordered_regions=True,manual_variance_sectors=True,variance_texture_cost=a.texture_cost,
            variance_spread=a.spread,coverage_layout='15ordered360_rgb200',trained=False),sub/'initial.pt')
        records.append(dict(wide=wide,check=check,full_frame_all_node_error=node_error,
            package_sha256=file_hash(sub/'coefficients-padded.bin')))
        print('VARIANCE_B1',records[-1],flush=True)
    board=Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14);by_id={r['id']:r for r in rows}
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        pics=[Image.open(by_id[key]['path']),Image.open(by_id[key]['target']),images[key,False],images[key,True]]
        for j,(im,title) in enumerate(zip(pics,('原图','区域目标','方差选区·真实B1','大窗口方差·真实B1'))):
            board.paste(im.crop(box),(j*420,i*314+34));draw.text((j*420+4,i*314+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'comparison.png');shutil.copy2(__file__,dest/Path(__file__).name)
    write_json(dest/'review.json',dict(records=records,trained=False,added_operators=0,postfilter=False,
        approximate_convex_variance=True,not_anisotropic_kuwahara=True,minimum_area_guaranteed=False,visual_acceptance=False,
        texture_cost=a.texture_cost,spread=a.spread,reference='https://www.kyprianidis.com/p/pg2009/'))


if __name__=='__main__':main()
