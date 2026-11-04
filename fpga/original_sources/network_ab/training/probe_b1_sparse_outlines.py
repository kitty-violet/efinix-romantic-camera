"""Draw only large region boundaries using one existing final feature channel.

Regroup8 luminance masks into3 monotone groups, freeing one of8 output
features for a thresholded Laplacian of the simplified luminance regions.
There is no source-detail edge detector or postprocessing stage.
"""
import argparse
import copy
import shutil
from pathlib import Path
import torch
from PIL import Image,ImageDraw,ImageFont
from . import train_b1_minimum_area as m
from .runtime import configure_workspace,write_json,file_hash
from .engine import set_reproducible


@torch.no_grad()
def outline_head(net,amount,threshold=6,retain_aa=False):
    aa=round(float(net.weight['up2_high_dw'][0,0].sum()*127));per_level=aa/2
    old=net.weight['rgb_head'].clone()
    kernel=net.weight['up2_high_dw'][0,0].clone() if retain_aa else old.new_tensor([[1,2,1],[2,4,2],[1,2,1]])
    kernel=kernel/kernel.sum()
    for key in ('up2_low_pw','up2_high_dw'):
        net.weight[key].zero_();net.bias[key].zero_()
    net.weight['rgb_head'].zero_()
    groups=[(0,3),(3,6),(6,8),(8,10),(10,12),(12,14),(14,16)]
    for c,(lo,hi) in enumerate(groups):
        net.weight['up2_low_pw'][c,lo:hi,0,0]=1/(hi-lo)
        net.weight['up2_high_dw'][c,0]=kernel*((hi-lo)*per_level)/127
    net.weight['rgb_head'][:,0]=(2*old[:,0]+old[:,1])/3
    net.weight['rgb_head'][:,1]=(old[:,1]+2*old[:,2])/3
    net.weight['rgb_head'][:,2]=old[:,3]
    net.weight['rgb_head'][:,3:7]=old[:,4:8]
    net.weight['up2_low_pw'][7,:8,0,0]=1/8
    lap=old.new_tensor([[0,-.25,0],[-.25,1,-.25],[0,-.25,0]])
    net.weight['up2_high_dw'][7,0]=lap;net.bias['up2_high_dw'][7]=-threshold
    net.weight['rgb_head'][:,7,0,0]=-amount
    for key in ('up2_low_pw','up2_high_dw','rgb_head'):
        w=net.weight[key];s=torch.pow(2,torch.ceil(torch.log2(w.abs().flatten(1).amax(1).clamp_min(1e-5)/127)))
        getattr(net,key+'_m').copy_((s*131072).clamp(1,131071).double()[None,:,None,None])


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--name',required=True)
    ap.add_argument('--threshold',type=float,default=6);ap.add_argument('--amounts',type=float,nargs=2,default=[1,2])
    ap.add_argument('--retain-aa',action='store_true');a=ap.parse_args()
    configure_workspace();set_reproducible(20261006183);dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    state=torch.load(a.warm,map_location='cpu',weights_only=True);net=m.t.r.initialize();net.load_state_dict(state['model']);net.quantized=True
    net.package['residual_encoding']=state['residual_encoding'];initial=copy.deepcopy(net.state_dict());d=m.load_data();images={r['id']:[] for r in d['validation']}
    for name,amount in [('current',None),('regroup',0),('outline1',a.amounts[0]),('outline2',a.amounts[1])]:
        net.load_state_dict(initial)
        if amount is not None:outline_head(net,amount,a.threshold,a.retain_aa)
        out=dest/name;out.mkdir();raw=net.export();(out/'coefficients-padded.bin').write_bytes(raw)
        actual=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw));records=[]
        for row in d['validation']:
            x=m.t.r.tensor(m.t.r.read(row['path']))[None]
            # Full node equality validates the actual package computation;
            # this probe still requires the independent scalar oracle if selected.
            ours=net(x,all_nodes=True);nodes,_=actual(x*128,True)
            err=max(float((v-nodes[k]).abs().max()) for k,v in ours.items());assert err==0
            image=m.t.r.c.rgb_image(nodes['rgb_head'][0]/128);image.save(out/(row['id']+'.png'));images[row['id']].append(image)
            edge=nodes['up2_high_dw'][:,7] if amount is not None else None
            records.append(dict(id=row['id'],node_error=err,outlined_fraction=float((edge>0).float().mean()) if edge is not None else None))
            del ours,nodes
        torch.save(dict(model=net.state_dict(),qat=True,step=state.get('step'),residual_encoding=state['residual_encoding'],
            parent=a.warm,coverage_layout='3Y_2Co_2Cg_1outline' if amount is not None else '4Y_2Co_2Cg',
            outline_amount=amount,outline_threshold=a.threshold,field_node=state.get('field_node'),
            fixed_sampling=state.get('fixed_sampling',False),classifier_mode=state.get('classifier_mode',False)),out/'initial.pt')
        write_json(out/'review.json',dict(records=records,added_operators=0,postfilter=False,visual_acceptance=False,independent_oracle_pending=True))
        print('SPARSE_OUTLINE',name,flush=True);del actual;torch.cuda.empty_cache()
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',15);board=Image.new('RGB',(1680,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board)
    for i,(key,box,label) in enumerate(m.t.r.c.CROPS):
        for j,title in enumerate(('当前网络','仅重排色层','仅大边界描线1','仅大边界描线2')):
            board.paste(images[key][j].crop(box),(420*j,314*i+34));draw.text((420*j+4,314*i+5),label+' | '+title,font=font,fill='white')
    board.save(dest/'comparison.png');shutil.copy2(__file__,dest/Path(__file__).name)
    write_json(dest/'review.json',dict(args=vars(a),parent=a.warm,parent_sha256=file_hash(a.warm),source_detail_edges=False,added_operators=0,visual_acceptance=False))


if __name__=='__main__':main()
