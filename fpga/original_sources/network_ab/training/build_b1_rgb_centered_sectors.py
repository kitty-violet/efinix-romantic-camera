"""Protect colour boundaries when choosing a flat neighbouring sector.

|candidate-centre| =2ReLU(candidate-centre)-candidate+centre needs only
31existing channels for12RGB candidates,3centre colours,4texture costs and
12positive differences. No new FPGA operator, plane or memory is introduced.
"""
import argparse
import shutil
from pathlib import Path
import torch
from . import train_b1_minimum_area as m
from .build_b1_centered_sectors import build as centered
from .runtime import configure_workspace,write_json,file_hash
from .publish_b1_style_recovery import export
from .probe_b1_decoder_coverage import fragment


@torch.no_grad()
def build(net,penalty,head,votes=True):
    centered(net,0,head)
    k='res1_conv1__pw';w,b=net.weight[k],net.bias[k];w.zero_();b.zero_()
    for c in range(12):w[c,c,0,0]=1
    for c in range(3):w[12+c,12+c,0,0]=1
    for s in range(4):
        w[15+s,15+s,0,0]=1
        for c in range(3):w[19+3*s+c,3*s+c,0,0]=1;w[19+3*s+c,12+c,0,0]=-1
    net.weight['res1_conv2__dw'].zero_();net.bias['res1_conv2__dw'].zero_()
    net.weight['res1_conv2__dw'][:31,0,1,1]=1
    k='res1_conv2__pw';w,b=net.weight[k],net.bias[k];w.zero_();b.zero_()
    for c in range(12):w[c,c,0,0]=1
    for s in range(4):
        w[12+s,15+s,0,0]=1
        w[12+s,19+3*s:22+3*s,0,0]=2*penalty/3
        w[12+s,3*s:3*s+3,0,0]=-penalty/3
        w[12+s,12:15,0,0]=penalty/3
    if not votes:
        for k,ids in [('res2_conv2__dw',[12,13]),('res3_conv2__dw',[6])]:
            for c in ids:net.weight[k][c].zero_();net.weight[k][c,0,1,1]=1;net.bias[k][c]=0
    for key in ('res1_conv1__pw','res1_conv2__dw','res1_conv2__pw'):
        w=net.weight[key];scale=torch.pow(2,torch.ceil(torch.log2(w.abs().flatten(1).amax(1).clamp_min(1e-5)/127)))
        getattr(net,key+'_m').copy_((scale*131072).clamp(1,131071).double()[None,:,None,None])
    return net


@torch.no_grad()
def align_routed_cost(net,votes=True):
    """Transport RGB and its cost with the same spatially voted selector.

The earlier heuristic transported pointwise min(costA,costB) even when the
spatial vote selected the other colour. The following comparison consequently
received a score belonging to a different candidate. Carry RGBC together.
"""
    keys=['res2_conv1__pw','res2_conv2__dw','res2_conv2__pw','res3_conv1__dw','res3_conv1__pw','res3_conv2__dw','res3_conv2__pw']
    for key in keys:net.weight[key].zero_();net.bias[key].zero_()
    w,b=net.weight['res2_conv1__pw'],net.bias['res2_conv1__pw']
    for c in range(16):w[c,c,0,0]=1
    for p in range(2):w[16+p,13+2*p,0,0]=127;w[16+p,12+2*p,0,0]=-127;b[16+p]=127
    net.weight['res2_conv2__dw'][:18,0,1,1]=1
    if votes:
        for c in (16,17):net.weight['res2_conv2__dw'][c,0]=1;net.bias['res2_conv2__dw'][c]=-4*127
    w,b=net.weight['res2_conv2__pw'],net.bias['res2_conv2__pw']
    for p in range(2):
        for side in range(2):
            sector=2*p+side
            for c in range(4):
                out=p*8+side*4+c;inp=3*sector+c if c<3 else 12+sector
                w[out,inp,0,0]=1;w[out,16+p,0,0]=1 if side==0 else -1
                b[out]=-127 if side==0 else 0
    net.weight['res3_conv1__dw'][:16,0,1,1]=1
    w,b=net.weight['res3_conv1__pw'],net.bias['res3_conv1__pw']
    for p in range(2):
        for c in range(4):w[p*4+c,p*8+c,0,0]=1;w[p*4+c,p*8+4+c,0,0]=1
    for p,sign in ((0,-1),(1,1)):
        for side in range(2):w[8,p*8+side*4+3,0,0]=127*sign
    b[8]=127;net.weight['res3_conv2__dw'][:9,0,1,1]=1
    if votes:net.weight['res3_conv2__dw'][8,0]=1;net.bias['res3_conv2__dw'][8]=-4*127
    w,b=net.weight['res3_conv2__pw'],net.bias['res3_conv2__pw']
    for c in range(3):
        w[c,c,0,0]=1;w[c,8,0,0]=1;b[c]=-127
        w[c+3,c+4,0,0]=1;w[c+3,8,0,0]=-1
    for key in keys:
        w=net.weight[key];scale=torch.pow(2,torch.ceil(torch.log2(w.abs().flatten(1).amax(1).clamp_min(1e-5)/127)))
        getattr(net,key+'_m').copy_((scale*131072).clamp(1,131071).double()[None,:,None,None])


@torch.no_grad()
def verify_routed_cost(net):
    device=net.weight['stem'].device;g=torch.Generator(device=device).manual_seed(815)
    z=torch.zeros(16,32,8,8,device=device);z[:,:16]=torch.randint(0,128,(16,16,8,8),device=device,generator=g)
    n=fragment(net,z,'res2_conv1.dw');out=n['res3_conv1.dw'];errors=[]
    for p in range(2):
        choose=n['res2_conv2.dw'][:,16+p:17+p]>0
        av=torch.cat((z[:,6*p:6*p+3],z[:,12+2*p:13+2*p]),1)
        bv=torch.cat((z[:,6*p+3:6*p+6],z[:,13+2*p:14+2*p]),1)
        expected=torch.where(choose,av,bv);actual=out[:,8*p:8*p+4]+out[:,8*p+4:8*p+8]
        errors.append(float((expected-actual).abs().max()))
    assert max(errors)==0,errors
    return dict(random_points=1024,selected_rgb_and_cost_error=max(errors))


@torch.no_grad()
def verify_cost(net,penalty):
    device=net.weight['stem'].device;g=torch.Generator(device=device).manual_seed(7103)
    z=torch.zeros(16,32,8,8,device=device);z[:,:15]=torch.randint(0,100,(16,15,8,8),device=device,generator=g)
    z[:,15:19]=torch.randint(0,10,(16,4,8,8),device=device,generator=g)
    p=fragment(net,z,'res1_conv1.dw')
    want=z[:,15:19]+penalty*(z[:,:12].reshape(16,4,3,8,8)-z[:,None,12:15]).abs().mean(2)
    got=p['res2_conv1.dw'][:,12:16];error=float((got-net.round(want).clamp(0,127)).abs().max())
    assert error==0,error
    # Equal brightness but distinct colour: scalar-Y protection cannot see this.
    centre=torch.tensor([80.,40.,40.],device=device)
    other=torch.tensor([40.,60.,40.],device=device)
    lum=torch.tensor([.25,.5,.25],device=device)
    assert torch.dot(centre,lum)==torch.dot(other,lum)
    return dict(integer_rgb_cost_error=error,random_spatial_values=1024,
                equal_luma_colour_distance=float((centre-other).abs().mean()))


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--warm',required=True)
    p.add_argument('--penalties',nargs='+',type=float,default=[.75,1.5]);p.add_argument('--no-votes',action='store_true')
    p.add_argument('--aligned-cost',action='store_true');a=p.parse_args()
    configure_workspace();torch.set_num_threads(2);dest=m.OUT/a.name;dest.mkdir(exist_ok=False);head=torch.load(a.warm,map_location='cpu',weights_only=True)['model']
    data=m.load_data();records=[]
    for penalty in a.penalties:
        net=m.t.r.initialize();net.quantized=True;build(net,penalty,head,not a.no_votes);check=verify_cost(net,penalty)
        if a.aligned_cost:align_routed_cost(net,not a.no_votes);check.update(verify_routed_cost(net))
        for row in data['validation']:
            nodes=net(m.t.r.tensor(m.t.r.read(row['path']))[None],all_nodes=True)
            candidates=nodes['res1_conv1.dw'][:,:12].reshape(1,4,3,180,320)
            picked=nodes['up1_low_dw'][:,:3]+nodes['up1_low_dw'][:,3:6]
            assert (candidates==picked[:,None]).all(2).any(1).all()
            del nodes
        sub=dest/str(penalty).replace('.','p');sub.mkdir();export(net,data['validation'],sub/'package');m.t.preview(net,data['validation'],sub/'preview')
        torch.save(dict(model=net.state_dict(),qat=True,step=0,residual_encoding=net.package['residual_encoding'],
            ordered_regions=True,rgb_centered_sectors=True,centre_penalty=penalty,trained=False),sub/'initial.pt')
        records.append(dict(penalty=penalty,check=check,one_candidate_selection=True));print('RGB_CENTERED',records[-1],flush=True)
    shutil.copy2(__file__,dest/Path(__file__).name)
    write_json(dest/'review.json',dict(args=vars(a),records=records,parent_head_sha256=file_hash(a.warm),code_sha256=file_hash(__file__),
        actual_B1=True,added_operators=0,postfilter=False,visual_acceptance=False,global_minimum_area_guaranteed=False))


if __name__=='__main__':main()
