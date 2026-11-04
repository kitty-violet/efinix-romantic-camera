"""16 exclusive full-RGB regions, refined at360p before RGB mixing.

The final eight channels need only three RGB channels. There is no constraint
pairing unrelated palette entries on common colour rays.
"""
import argparse
import json
import shutil
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .probe_b1_decoder_coverage import fragment
from .runtime import configure_workspace,write_json,file_hash
from .b1_trained_image import runtime_from_image
from .probe_b1_cartoon_int8 import IntegerB1


@torch.no_grad()
def build(net,palette):
    start=False;keys=[]
    for key in net.weight:
        if key=='res1_conv1__pw':start=True
        if start:net.weight[key].zero_();net.bias[key].zero_();keys.append(key)
    net.bias['res1_conv1__pw'][:16]=64
    kernel=net.weight['stem'].new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
    net.weight['res1_conv2__dw'][:16,0]=kernel/18
    for stage in (1,2,3):net.package['residual_encoding'][f'res{stage}_add']=dict(main_scaled=False,multiplier=1,post_multiplier=131071)
    # First pair reduction also stores all original encoded scores.
    k='res1_conv2__pw'
    for j in range(8):
        a,b=2*j,2*j+1;net.weight[k][j,a,0,0]=16;net.weight[k][j,b,0,0]=-16;net.bias[k][j]=a-b
        net.weight[k][j+8,b,0,0]=16;net.bias[k][j+8]=b
    for j in range(16):net.weight[k][j+16,j,0,0]=16;net.bias[k][j+16]=j
    for k in ('res2_conv1__dw','res2_conv2__dw','res3_conv1__dw','res3_conv2__dw','up1_low_dw'):
        net.weight[k][:,0,1,1]=1
    # (number of input maxima, start of their second terms, score offset)
    for key,count,base,score,newscore in [('res2_conv1__pw',8,8,16,8),('res2_conv2__pw',4,4,8,4),('res3_conv1__pw',2,2,4,2)]:
        half=count//2
        for j in range(half):
            a,b=2*j,2*j+1
            for v in (a,a+base):net.weight[key][j,v,0,0]=1
            for v in (b,b+base):net.weight[key][j,v,0,0]=-1;net.weight[key][j+half,v,0,0]=1
        for j in range(16):net.weight[key][newscore+j,score+j,0,0]=1
    k='res3_conv2__pw'
    for j in range(16):
        net.weight[k][j,j+2,0,0]=127;net.weight[k][j,:2,0,0]=-127;net.bias[k][j]=127
        net.weight['up1_low_pw'][j,j,0,0]=1
        net.weight['up1_high_dw'][j,0]=kernel
        net.weight['up2_low_dw'][j,0]=3*kernel;net.bias['up2_low_dw'][j]=-127
    net.weight['up2_low_pw'][:3,:16,0,0]=palette.T/254
    for c in range(3):
        net.weight['up2_high_dw'][c,0]=.5*kernel;net.weight['up2_high_dw'][c,0,1,1]+=.5
        net.weight['rgb_head'][c,c,0,0]=2
    net.bias['rgb_head'].fill_(-128)
    for k in keys:
        w=net.weight[k];s=torch.pow(2,torch.ceil(torch.log2(w.abs().flatten(1).amax(1).clamp_min(1e-5)/127)))
        getattr(net,k+'_m').copy_((s*131072).clamp(1,131071).double()[None,:,None,None])


@torch.no_grad()
def palette_actual(net):
    z=torch.zeros(16,16,1,1,device='cuda')
    for j in range(16):z[j,j]=127
    return fragment(net,z,'up1_low_pw')['rgb_head'][:,:,0,0]/128


@torch.no_grad()
def coverage(target,palette):
    flat=target.permute(0,2,3,1).reshape(-1,3);ids=[]
    for first in range(0,len(flat),8192):
        x=flat[first:first+8192];ids.append((x.square().sum(1,keepdim=True)+palette.square().sum(1)[None]-2*x@palette.T).argmin(1))
    ids=torch.cat(ids).reshape(target.shape[0],*target.shape[2:])
    return F.avg_pool2d(F.one_hot(ids,16).permute(0,3,1,2).float(),4)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--palette',required=True);ap.add_argument('--name',required=True);a=ap.parse_args()
    configure_workspace();torch.set_num_threads(2);dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    state=torch.load(a.warm,map_location='cpu',weights_only=True);net=m.t.r.initialize();net.load_state_dict(state['model']);net.quantized=True
    pal=torch.tensor(json.loads(Path(a.palette).read_text())['palette'],device='cuda');build(net,pal)
    data=m.load_data();sample=m.t.r.tensor(m.t.r.read(data['validation'][0]['path']))[None,...,:64,:64]
    full=net(sample,all_nodes=True);part=fragment(net,full['res1_conv2.dw'],'res1_conv2.dw')
    err=max(float((v-full[k]).abs().max()) for k,v in part.items() if k not in ('res1_conv2.dw',));assert err==0
    raw=net.export();actual=IntegerB1(runtime_from_image(raw));got,_=actual(sample*128,True)
    actual_error=max(float((v-got[k]).abs().max()) for k,v in full.items());assert actual_error==0
    # Random bounded class scores test all16 priorities, independent of images.
    z=torch.zeros(128,32,4,4,device='cuda');z[:,:16]=torch.randint(0,8,(128,16,4,4),device='cuda')
    out=fragment(net,z,'res1_conv2.dw');encoded=z[:,:16]*16+torch.arange(16,device='cuda')[None,:,None,None]
    expected=F.one_hot(encoded.argmax(1),16).permute(0,3,1,2)*127
    winner_error=float((out['up1_low_pw']-expected).abs().max());assert winner_error==0,winner_error
    palette=palette_actual(net);rows={r['id']:r for r in data['validation']};records=[]
    board=Image.new('RGB',(1260,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',14)
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        target=m.t.r.tensor(m.t.r.read(rows[key]['target'])[box[1]:box[3],box[0]:box[2]])[None]
        cov=coverage(target,palette);q=fragment(net,(cov*127).round(),'up1_low_pw')['rgb_head']/128
        images=[m.t.r.c.rgb_image(target[0]),m.t.r.c.rgb_image(q[0])]
        for j,(im,title) in enumerate(zip(images,('离线目标','16色mask注入诊断·非网络预测'))):
            board.paste(im,(j*420,i*314+34));draw.text((j*420+4,i*314+5),label+' | '+title,font=font,fill='white')
        records.append(dict(crop=label,mae=float((q-target).abs().mean()*128)))
    board.crop((0,0,840,board.height)).save(dest/'diagnostic.png')
    torch.save(dict(model=net.state_dict(),qat=True,step=0,residual_encoding=net.package['residual_encoding'],
        coverage_layout='16exclusive_rgb',sixteen_winner=True,decoder_diagnostic_only=True,palette_source=a.palette,parent=a.warm),dest/'initial.pt')
    shutil.copy2(__file__,dest/Path(__file__).name)
    write_json(dest/'results.json',dict(small_all_node_error=actual_error,fragment_error=err,random_winner_cases=2048,winner_error=winner_error,
        records=records,target_injection_only=True,network_trained=False,visual_acceptance=False,palette=palette.cpu().tolist(),added_operators=0))
    print('SIXTEEN_DECODER_PASS',records,flush=True)


if __name__=='__main__':main()
