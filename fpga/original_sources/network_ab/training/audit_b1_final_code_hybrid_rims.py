"""Measure hybrid-code bright rims using known endpoint labels as diagnosis."""
import argparse,json,shutil
from pathlib import Path
import cv2,numpy as np,torch
from .probe_b1_encoder_gauge import load
from .probe_b1_decoder_coverage import fragment
from .runtime import write_json,file_hash
from . import train_b1_minimum_area as m

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','mining','out'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();torch.set_num_threads(2);cv2.setNumThreads(2);out=Path(a.out);out.mkdir(exist_ok=False)
    state,net=load(a.checkpoint);code=torch.arange(256);bits=((code[:,None]//(2**torch.arange(8)[None]))%2).float()
    palette=fragment(net,(bits*127)[:,:,None,None],'up2_high_dw')['rgb_head'][:,:,0,0]+128
    rows={r['id']:r for r in m.load_data(a.data)['training']};specs=json.loads(Path(a.mining).read_text('utf-8'))['records'];records=[]
    for ix in (1,5,6):
        spec=specs[ix];row=rows[spec['row']['id']];xx,yy=spec['xy'];w,h=spec['wh'];sl=np.s_[yy:yy+h,xx:xx+w]
        x=torch.from_numpy(m.t.r.read(row['path'])[sl].transpose(2,0,1).copy())[None].float()/128-1
        hard=m.t.r.read(row['hard_region_target'])[sl].astype(np.float32);lab=m.load_region(row['region'])[sl].astype(np.int32)
        uniq,inv=np.unique(lab,return_inverse=True);lab=inv.reshape(lab.shape);cnt=np.bincount(lab.ravel()).clip(1)
        colors=np.stack([np.bincount(lab.ravel(),weights=hard[...,j].ravel())/cnt for j in range(3)],1)
        choice=((torch.from_numpy(colors).float()[:,None]-palette[None])**2).mean(-1).argmin(-1).numpy()
        coded=choice[lab];lo=cv2.erode(coded.astype(np.uint16),np.ones((5,5),np.uint8)).astype(int);hi=cv2.dilate(coded.astype(np.uint16),np.ones((5,5),np.uint8)).astype(int)
        pad=np.pad(coded,2,mode='edge');other=np.zeros(lab.shape,bool)
        for dy in range(5):
            for dx in range(5):
                z=pad[dy:dy+h,dx:dx+w];other|=(z!=lo)&(z!=hi)
        aa=palette[lo].numpy();bb=palette[hi].numpy();valid=(lo!=hi)&~other&(np.linalg.norm(bb-aa,axis=-1)>=16)
        valid[:40]=False;valid[-40:]=False;valid[:,:40]=False;valid[:,-40:]=False
        nodes=net(x,all_nodes=True);p=nodes['rgb_head'][0].permute(1,2,0).numpy()+128
        bpred=(nodes['up2_high_dw'][0].permute(1,2,0).numpy()>=64);ba=bits[lo].numpy().astype(bool);bc=bits[hi].numpy().astype(bool)
        commonok=np.all((ba!=bc)|(bpred==ba),axis=-1)
        mixed=commonok&np.any(bpred!=ba,axis=-1)&np.any(bpred!=bc,axis=-1)
        y=lambda z:(z[...,0]+2*z[...,1]+z[...,2])/4
        white=(y(p)>np.maximum(y(aa),y(bb))+4)&valid
        weights=2**np.arange(8);hybrid=palette[(bpred*weights).sum(-1)].numpy()
        hybridwhite=(y(hybrid)>np.maximum(y(aa),y(bb))+4)&valid
        rec=dict(id=row['id'],two_code_boundary_pixels=int(valid.sum()),actual_white_pixels=int(white.sum()),
            actual_white_fraction=float(white.sum()/max(1,valid.sum())),
            white_with_legal_endpoint_bit_hybrid=int((white&mixed).sum()),
            white_with_common_bit_error=int((white&~commonok).sum()),
            thresholded_hybrid_white_pixels=int(hybridwhite.sum()))
        records.append(rec);print('CODE_HYBRID_RIMS',rec,flush=True)
    shutil.copy2(__file__,out/Path(__file__).name)
    write_json(out/'review.json',dict(records=records,checkpoint_sha256=file_hash(a.checkpoint),
        diagnostic_only=True,mask='Exactly2 nearest palette codes in5x5, contrast>=16; TRAIN central240',
        caveat='Endpoint labels are teacher-derived; this partitions local errors, not all visual defects.'))

if __name__=='__main__':main()
