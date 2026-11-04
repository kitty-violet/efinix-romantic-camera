"""Fit only the existing RGB head on21 TRAIN frames; keep geometry frozen."""
import argparse,copy,shutil
from pathlib import Path
import torch
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .runtime import write_json,file_hash,configure_workspace
from . import train_b1_minimum_area as m

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','out'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();configure_workspace();torch.set_num_threads(2);out=Path(a.out);out.mkdir(exist_ok=False);state,net=load(a.checkpoint);net.cuda();data=m.load_data(a.data)
    rows=[]
    for vid in sorted({r['video'] for r in data['training']}):
        rr=[r for r in data['training'] if r['video']==vid];rows.extend(rr[i] for i in sorted(set((0,len(rr)//2,len(rr)-1))))
    xx=[];yy=[];hashes={}
    for row in rows:
        x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float().cuda()/128-1
        z=net(x,all_nodes=True)['up2_high_dw'][0,:,::8,::8].flatten(1).T/127
        t=torch.from_numpy(m.t.r.read(row['target'])[::8,::8].copy()).float().cuda().reshape(-1,3)-128
        xx.append(torch.cat((z,torch.ones_like(z[:,:1])),1));yy.append(t);hashes[row['id']]=dict(source=file_hash(row['path']),target=file_hash(row['target']))
    X=torch.cat(xx).double();Y=torch.cat(yy).double();old=torch.cat((net.weight['rgb_head'][:,:,0,0].T*127,net.bias['rgb_head'][None]),0).double()
    diag=torch.eye(9,device='cuda',dtype=torch.float64)*.001;diag[-1,-1]=0
    solution=torch.linalg.solve(X.T@X/len(X)+diag,X.T@Y/len(X)+diag@old)
    before=float((X@old-Y).abs().mean());after=float((X@solution-Y).abs().mean());assert torch.isfinite(solution).all()
    # A fitted contrast reversal would undo the ordered level interpretation.
    assert (solution[:4]>0).all(),solution[:4]
    base=copy.deepcopy(net);net.weight['rgb_head'].copy_(solution[:8].T[:,:,None,None].float()/127);net.bias['rgb_head'].copy_(solution[-1].float())
    w=net.weight['rgb_head'];mult=(w.abs().flatten(1).amax(1)/126*131072).ceil().clamp_min(1)
    net.rgb_head_m.copy_(mult.double()[None,:,None,None]);scale=mult[:,None,None,None]/131072;w.copy_(net.round(w/scale)*scale)
    for key,value in base.state_dict().items():
        if 'rgb_head' not in key:assert torch.equal(value,net.state_dict()[key]),key
    images={};records=[]
    for name,model in (('before',base),('calibrated',net)):
        sub=out/name;sub.mkdir();raw=model.export();assert len(raw)==20480;(sub/'coefficients-padded.bin').write_bytes(raw);rt=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cuda')
        for row in data['validation']:
            x=torch.from_numpy(m.t.r.read(row['path']).transpose(2,0,1).copy())[None].float().cuda()/128-1
            nodes=model(x,all_nodes=True);actual,_=rt(model.round(x*128),True);assert all(torch.equal(v,actual[k]) for k,v in nodes.items())
            im=m.t.r.c.rgb_image(actual['rgb_head'][0]/128);im.save(sub/(row['id']+'_output.png'));images[name,row['id']]=im;records.append(dict(mode=name,id=row['id'],all_nodes_error=0))
        saved=copy.deepcopy(state);saved.update(model={k:v.cpu() for k,v in model.state_dict().items()},residual_encoding=model.package['residual_encoding'],palette_calibrated=name=='calibrated',geometry_frozen=True,visual_acceptance=False);torch.save(saved,sub/'candidate.pt')
    byid={r['id']:r for r in data['validation']};board=Image.new('RGB',(1260,314*len(m.t.r.c.CROPS)),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12)
    for i,(rid,box,label) in enumerate(m.t.r.c.CROPS):
        ims=[('固定目标',Image.open(byid[rid]['target']))]+[(k,images[k,rid]) for k in ('before','calibrated')]
        for j,(title,im) in enumerate(ims):board.paste(im.crop(box),(420*j,314*i+34));draw.text((420*j+3,314*i+4),label+' | '+title,font=font,fill='white')
    board.save(out/'comparison.png');shutil.copy2(__file__,out/Path(__file__).name);write_json(out/'review.json',dict(args=vars(a),source_sha256=file_hash(__file__),training_rows=hashes,head_before=old.cpu().tolist(),head_after=solution.cpu().tolist(),train_mae_before=before,train_mae_after=after,records=records,geometry_frozen=True,extra_inference_ops=0,visual_acceptance=False));print('PALETTE_CALIBRATION',before,after,'TRAIN',len(rows),flush=True)

if __name__=='__main__':main()
