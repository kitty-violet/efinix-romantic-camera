"""Exactly three requested revisions, same B1 graph/data/colour calibration.

A: classify colours at360 before second decimation; retain occupancies.
B: A plus one3x3 majority cleanup at180.
C: A plus centre5 majority, preserving ideal one-pixel lines at180.
No inference postfilter, new operator, or train/validation leakage in head fit.
"""
import argparse,copy,json,shutil
from pathlib import Path
import torch
from PIL import Image,ImageDraw,ImageFont
from .design_b1_joint_colour_levels import install as levels
from .design_b1_edge_stopping_diffusion import clear,identity_dw,KERNEL
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .runtime import configure_workspace,write_json,file_hash
from .publish_b1_style_recovery import export
from . import train_b1_minimum_area as m

@torch.no_grad()
def install(net,mode):
    levels(net,2)
    # Rebuild the whole front explicitly. The ordered225-colour tail stays.
    for k in net.weight:
        if k=='up1_high_dw':break
        clear(net,k)
    for c in range(3):net.weight['stem'][c,c,1,1]=.5;net.bias['stem'][c]=64
    identity_dw(net,'down1__dw',range(3),KERNEL)
    k='down1__pw';clear(net,k,.5)
    fields=[(torch.tensor([.25,.5,.25]),0,tuple(11+14*j for j in range(8))),
            (torch.tensor([.5,0.,-.5]),64,(52,60,68,76)),
            (torch.tensor([-.25,.5,-.25]),64,(52,60,68,76))]
    c=0
    for coef,bias,thresholds in fields:
        for threshold in thresholds:
            net.weight[k][c,:3,0,0]=64*coef;net.bias[k][c]=63.5+64*(bias-threshold);c+=1
    identity_dw(net,'down2__dw',range(16),KERNEL)
    for k in ('down2__pw','res1_conv1__pw','res1_conv2__pw','res2_conv1__pw','res2_conv2__pw','res3_conv1__pw','res3_conv2__pw','up1_low_pw'):
        for c in range(16):net.weight[k][c,c,0,0]=1
    for k in ('res1_conv1__dw','res1_conv2__dw','res2_conv1__dw','res2_conv2__dw','res3_conv1__dw','res3_conv2__dw','up1_low_dw'):identity_dw(net,k,range(16))
    if mode!='A':
        k='res1_conv1__dw';clear(net,k,1/16)
        kernel=torch.ones(3,3);kernel[1,1]=5 if mode=='C' else 1
        for c in range(16):net.weight[k][c,0]=kernel;net.bias[k][c]=-(6 if mode=='C' else 4)*127
    for r in range(1,4):net.package['residual_encoding'][f'res{r}_add']=dict(main_scaled=False,multiplier=1,post_multiplier=131071)
    for k,w in net.weight.items():
        s=(getattr(net,k+'_m')/131072).float().flatten()[:,None,None,None];assert (w/s).abs().max()<=127.001,(k,float((w/s).abs().max()))
        w.copy_(net.round(w/s)*s)

@torch.no_grad()
def regression(net,mode):
    yy,xx=torch.meshgrid(torch.arange(17),torch.arange(17),indexing='ij');device=next(net.parameters()).device
    patterns={'dot':(xx==8)&(yy==8),'vertical':xx==8,'horizontal':yy==8,'diagonal':xx==yy,'corner':(xx>=8)&(yy>=8)};result={}
    for name,mask in patterns.items():
        z=torch.zeros(1,32,17,17,device=device);z[:,:16]=mask.to(device)*127
        p=fragment(net,z,'down2.pw')['res1_add'][:,0];centre=float(p[0,8,8]);result[name]=centre
        expected=127 if mode=='A' or (mode=='C' and name!='dot') else 0
        assert centre==expected,(mode,name,centre,expected)
    # Positive geometry must preserve nesting of all ordered predicates.
    x=torch.randint(-128,128,(1,3,32,40),device=device).float()/128;nodes=net(x,all_nodes=True)
    for key in ('down1.pw','down2.pw','res3_add','up1_high_dw','up2_low_dw'):
        for lo,hi in ((0,8),(8,12),(12,16)):assert (torch.diff(nodes[key][:,lo:hi],dim=1)<=0).all(),key
    return dict(ideal180_core=result,ordered_predicates=True,not_a720_thin_line_guarantee=True)

@torch.no_grad()
def calibrate(net,rows):
    xx=[];yy=[]
    for i,row in enumerate(rows):
        assert file_hash(row['path'])==row['sha256'] and file_hash(row['target'])==row['target_sha256']
        x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float().cuda()/128-1
        z=net(x,all_nodes=True)['up2_high_dw'][0,:,::8,::8].flatten(1).T/127
        t=torch.from_numpy(m.t.r.read(row['target'])[::8,::8].copy()).float().cuda().reshape(-1,3)-128
        xx.append(torch.cat((z,torch.ones_like(z[:,:1])),1));yy.append(t)
        if (i+1)%7==0:print('THREE_REVISIONS_CALIBRATION',i+1,len(rows),flush=True)
    X=torch.cat(xx).double();Y=torch.cat(yy).double();old=torch.cat((net.weight['rgb_head'][:,:,0,0].T*127,net.bias['rgb_head'][None]),0).double()
    diag=torch.eye(9,device='cuda',dtype=torch.float64)*.001;diag[-1,-1]=0
    h=torch.linalg.solve(X.T@X/len(X)+diag,X.T@Y/len(X)+diag@old);assert (h[:4]>0).all() and torch.isfinite(h).all()
    before=float((X@old-Y).abs().mean());after=float((X@h-Y).abs().mean())
    net.weight['rgb_head'].copy_(h[:8].T[:,:,None,None].float()/127);net.bias['rgb_head'].copy_(h[-1].float())
    w=net.weight['rgb_head'];mult=(w.abs().flatten(1).amax(1)/126*131072).ceil().clamp_min(1)
    net.rgb_head_m.copy_(mult.double()[None,:,None,None]);scale=mult[:,None,None,None]/131072;w.copy_(net.round(w/scale)*scale)
    return dict(train_mae_before=before,train_mae_after=after,train_ids=[r['id'] for r in rows],head=h.cpu().tolist())

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','baseline','data','out'):ap.add_argument('--'+k,required=True)
    ap.add_argument('--resume',action='store_true')
    a=ap.parse_args();configure_workspace();torch.set_num_threads(2);out=Path(a.out);out.mkdir(exist_ok=a.resume);state,base=load(a.checkpoint);data=m.load_data(a.data);images={};records=[];review={}
    if a.resume:review=json.loads((out/'progress.json').read_text('utf-8'))
    rows=[]
    for vid in sorted({r['video'] for r in data['training']}):
        rr=[r for r in data['training'] if r['video']==vid];rows.extend(rr[i] for i in sorted(set((0,len(rr)//2,len(rr)-1))))
    for mode in ('A','B','C'):
        if mode in review:
            assert file_hash(out/mode/'package/coefficients-padded.bin')==review[mode]['package_sha256']
            for row in data['validation']:images[mode,row['id']]=Image.open(out/mode/(row['id']+'_output.png')).copy()
            continue
        net=copy.deepcopy(base);install(net,mode);net.cuda();checks=regression(net,mode);calibration=calibrate(net,rows);sub=out/mode;sub.mkdir()
        export(net,data['validation'],sub/'package')
        for row in data['validation']:
            x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float().cuda()/128-1
            im=m.t.r.c.rgb_image(net(x)[0][0]);im.save(sub/(row['id']+'_output.png'));images[mode,row['id']]=im
        torch.save(dict(model={k:v.cpu() for k,v in net.state_dict().items()},residual_encoding=net.package['residual_encoding'],contract_sha256=state['contract_sha256'],designed_not_trained=True,palette_calibrated=True,geometry_frozen=True,visual_acceptance=False,extra_inference_ops=0,revision=mode),sub/'candidate.pt')
        review[mode]=dict(regression=checks,calibration=calibration,package_sha256=file_hash(sub/'package/coefficients-padded.bin'));write_json(out/'progress.json',review);print('THREE_REVISION_COMPLETE',mode,flush=True)
    byid={r['id']:r for r in data['validation']};board=Image.new('RGB',(2100,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
    for i,(rid,box,label) in enumerate(m.t.r.c.CROPS):
        ims=[('目标',Image.open(byid[rid]['target'])),('当前版',Image.open(Path(a.baseline)/(rid+'_output.png')))]+[(k,images[k,rid]) for k in ('A','B','C')]
        for j,(title,im) in enumerate(ims):board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+3,314*i+4),label+' | '+title,font=font,fill='white')
    board.save(out/'comparison.png');shutil.copy2(__file__,out/Path(__file__).name);write_json(out/'review.json',dict(args=vars(a),versions=review,source_sha256=file_hash(__file__),data_sha256=file_hash(a.data),exactly_three_revisions=True,visual_acceptance=False,extra_inference_ops=0))

if __name__=='__main__':main()
