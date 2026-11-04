"""Four PW layers implement8-way max+selection, leaving res1 learnable.

max(a,b)=ReLU(a-b)+b; adjacent reductions are fused into the next
comparison affine. This keeps one full32-channel residual for learned
spatial features, within the unchanged B1 operator/weight limits.
"""
import argparse
import shutil
from pathlib import Path
import torch
from . import train_b1_minimum_area as m
from .runtime import configure_workspace,write_json,file_hash
from .publish_b1_style_recovery import export
from .engine import set_reproducible


@torch.no_grad()
def compact(net):
    changed=[]
    # Existing bounded-score checkpoint: undo division at the early sampler.
    net.weight['down2__dw'][:8]*=8.5;changed.append('down2__dw')
    for j in range(9,16):
        net.weight['down1__pw'][j].zero_();net.weight['down1__pw'][j,(j-9)%8,0,0]=1
        net.bias['down1__pw'][j]=0
        net.weight['down2__dw'][j]=net.weight['down2__dw'][8]
    changed.append('down1__pw')
    key='down2__pw';net.weight[key].zero_();net.bias[key].zero_();changed.append(key)
    for j in range(32):
        src=j if j<16 else (8 if j==16 else 9+(j-17)%7)
        net.weight[key][j,src,0,0]=1
    kernel=net.weight['stem'].new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
    for stage in (1,2,3):
        net.package['residual_encoding'][f'res{stage}_add']=dict(main_scaled=False,multiplier=1,post_multiplier=131071)
        for half in (1,2):
            for kind in ('dw','pw'):
                k=f'res{stage}_conv{half}__{kind}';net.weight[k].zero_();net.bias[k].zero_();changed.append(k)
                if kind=='dw':net.weight[k][:,0,1,1]=1
    for half in (1,2):
        for j in range(32):net.weight[f'res1_conv{half}__pw'][j,j,0,0]=1
    k='res2_conv1__dw';net.weight[k][:8,0]=kernel/8.5;net.weight[k][16,0]=kernel
    k='res2_conv1__pw'
    for j in range(4):
        a,b=2*j,2*j+1
        net.weight[k][j,a,0,0]=8;net.weight[k][j,b,0,0]=-8;net.bias[k][j]=a-b
        net.weight[k][j+4,b,0,0]=8;net.bias[k][j+4]=b
    for j in range(8):net.weight[k][j+8,j,0,0]=8;net.bias[k][j+8]=j
    net.weight[k][16,16,0,0]=1
    k='res2_conv2__pw'
    for j in range(2):
        a,b=2*j,2*j+1
        for q in (a,a+4):net.weight[k][2*j,q,0,0]=1
        for q in (b,b+4):
            net.weight[k][2*j,q,0,0]=-1;net.weight[k][2*j+1,q,0,0]=1
    for j in range(8,17):net.weight[k][j,j,0,0]=1
    k='res3_conv1__pw'
    net.weight[k][0,0:2,0,0]=1;net.weight[k][0,2:4,0,0]=-1
    net.weight[k][1,2:4,0,0]=1
    for j in range(8,17):net.weight[k][j,j,0,0]=1
    k='res3_conv2__pw'
    for j in range(8):
        for out in (j,j+8):
            net.weight[k][out,j+8,0,0]=127;net.weight[k][out,:2,0,0]=-127
        net.bias[k][j]=127;net.weight[k][j+8,16,0,0]=1
    for k in ('up1_low_dw','up1_low_pw'):
        net.weight[k].zero_();net.bias[k].zero_();changed.append(k)
    net.weight['up1_low_dw'][:,0,1,1]=1
    for j in range(16):net.weight['up1_low_pw'][j,j,0,0]=1
    for k in set(changed):
        w=net.weight[k];s=torch.pow(2,torch.ceil(torch.log2(w.abs().flatten(1).amax(1).clamp_min(1e-5)/127)))
        getattr(net,k+'_m').copy_((s*131072).clamp(1,131071).double()[None,:,None,None])


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--warm',required=True);ap.add_argument('--name',required=True);a=ap.parse_args()
    configure_workspace();set_reproducible(20261006313);torch.set_num_threads(2);dest=m.OUT/a.name;dest.mkdir(exist_ok=False)
    state=torch.load(a.warm,map_location='cpu',weights_only=True);assert state.get('bounded_scores')
    net=m.t.r.initialize();net.load_state_dict(state['model']);net.package['residual_encoding']=state['residual_encoding'];net.quantized=True
    compact(net);d=m.load_data();checks=[]
    for row in d['validation']:
        n=net(m.t.r.tensor(m.t.r.read(row['path']))[None],all_nodes=True)
        scores=n['res2_conv1.dw'][:,:8];encoded=scores*8+torch.arange(8,device='cuda')[None,:,None,None]
        expected=F_one_hot(encoded.argmax(1),8).permute(0,3,1,2)*127
        # res3's final PW and residual are signed; the existing next DW
        # applies ReLU. Compare only after that specified activation.
        masks=n['up1_low_dw'][:,:8]
        error=float((masks-expected).abs().max());assert error==0,error
        checks.append(dict(id=row['id'],winner_error=error,score_max=float(scores.max())))
    export(net,d['validation'],dest/'package');m.t.preview(net,d['validation'],dest/'preview')
    torch.save(dict(model=net.state_dict(),qat=True,step=0,residual_encoding=net.package['residual_encoding'],
        coverage_layout='8exclusive_winner_rays',soft_gate=True,winner_shape=True,bounded_scores=True,
        compact_winner=True,parent=a.warm),dest/'initial.pt')
    shutil.copy2(__file__,dest/Path(__file__).name)
    write_json(dest/'review.json',dict(checks=checks,source_sha256=file_hash(a.warm),added_operators=0,
        source_inference=True,postfilter=False,visual_acceptance=False,minimum_area_guaranteed=False))
    print('COMPACT_WINNER_PASS',checks,flush=True)


from torch.nn.functional import one_hot as F_one_hot
if __name__=='__main__':main()
