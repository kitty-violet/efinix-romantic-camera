"""Fit an ordered colour head onto the clear learned sampling features.

Only existing DW/PW coefficients are used. Region labels are training-only.
"""
import torch
from torch.nn import functional as F
from . import train_b1_style_recovery as t
from .runtime import configure_workspace,write_json,file_hash
from .engine import set_reproducible

BASES=(8,24,40,56,72,92)
INCREMENTS=(16,16,16,16,20,24)


def labels(target):
    z=(target+1)*64;r,g,b=z[:,0:1],z[:,1:2],z[:,2:3]
    y=(r+2*g+b)/4;planes=[]
    for lo,inc in zip(BASES,INCREMENTS):
        binary=torch.cat([(y>lo+inc*(j+.5)/3).float() for j in range(3)],1)
        binary=(F.avg_pool2d(F.pad(binary,(1,1,1,1),mode='replicate'),3,1)>=.5).float()
        planes.append(binary.sum(1,keepdim=True)*(127/3))
    return torch.cat(planes+[(r-b)/2+64,(-r+2*g-b)/4+64],1)


@torch.inference_mode()
def main():
    configure_workspace();set_reproducible(2026100688)
    dest=t.OUT/'clear-region-initialization';dest.mkdir(exist_ok=False)
    parent=t.OUT/'structured-calibration/target_112.pt'
    state=torch.load(parent,map_location='cpu',weights_only=True);net=t.r.initialize()
    net.load_state_dict(state['model']);net.quantized=True
    d=t.data();rows=d['training'][::max(1,len(d['training'])//24)][:24]
    xx=torch.zeros(17,17,device='cuda',dtype=torch.float64);xy=torch.zeros(17,3,device='cuda',dtype=torch.float64)
    for row in rows:
        x=t.r.tensor(t.r.read(row['path']))[None];target=t.r.tensor(t.r.read(row['target']))[None]
        feat=net(x,all_nodes=True)['up2_low_dw'];z=F.avg_pool2d((target+1)*64,2)
        rr,gg,bb=z[:,0:1],z[:,1:2],z[:,2:3]
        colour=torch.cat([(rr+2*gg+bb)/4,(rr-bb)/2+64,(-rr+2*gg-bb)/4+64],1)
        a=feat[0,:,::4,::4].flatten(1).T.double()/64
        a=torch.cat([a,torch.ones_like(a[:,:1])],1)
        b=colour[0,:,::4,::4].flatten(1).T.double()
        xx+=a.T@a;xy+=a.T@b
    reg=torch.eye(17,device='cuda',dtype=torch.float64)*xx.diag().mean()*.00001;reg[-1,-1]=0
    fit=torch.linalg.solve(xx+reg,xy).float()
    w=net.weight['up2_low_pw'];bias=net.bias['up2_low_pw'];w.zero_()
    for c,(lo,inc) in enumerate(zip(BASES,INCREMENTS)):
        w[c,:,0,0]=fit[:-1,0]/64*(127/inc);bias[c]=(fit[-1,0]-lo)*(127/inc)
    for c in [6,7]:w[c,:,0,0]=fit[:-1,c-5]/64;bias[c]=fit[-1,c-5]
    kernel=w.new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
    for c in range(8):net.weight['up2_high_dw'][c,0]=kernel*(3/127 if c<6 else .5)
    net.bias['up2_high_dw'].zero_()
    h=net.weight['rgb_head'];h.zero_()
    for c,inc in enumerate(INCREMENTS):h[:,c,0,0]=2*inc/3
    h[:,6,0,0]=w.new_tensor([4,0,-4]);h[:,7,0,0]=w.new_tensor([-4,4,-4])
    net.bias['rgb_head'].copy_(w.new_tensor([-112,-240,144]))
    mm=(w.abs().flatten(1).amax(1).clamp_min(1e-5)*131072/112).ceil().clamp(1,131071)
    net.up2_low_pw_m.copy_(mm.double()[None,:,None,None])
    net.up2_high_dw_m[:,:6].fill_(32);net.up2_high_dw_m[:,6:].fill_(1024);net.rgb_head_m.fill_(17476)
    raw=net.export();(dest/'initial.bin').write_bytes(raw)
    actual=t.r.IntegerB1(t.r.runtime_from_image(raw));errors=[];ranges=[]
    for row in d['validation']:
        x=t.r.tensor(t.r.read(row['path']))[None];nodes=net(x,all_nodes=True);other,_=actual(x*128,True)
        errors.append(max(float((v-other[k]).abs().max()) for k,v in nodes.items()))
        a=nodes['up2_high_dw'][:,:6];ranges.append([float(a.min()),float(a.max())])
    assert max(errors)==0 and all(a==0 and b<=3 for a,b in ranges)
    torch.save(dict(model=net.state_dict(),qat=True,step=0,contract_sha256=t.r.CONTRACT_SHA),dest/'initial.pt')
    t.preview(net,d['validation'],dest/'preview')
    write_json(dest/'review.json',dict(parent=str(parent),parent_sha256=file_hash(parent),
        fit_training_rows=[r['id'] for r in rows],validation=t.evaluate(net,d['validation']),
        byte_all_node_error=max(errors),region_ranges=ranges,added_operators=0,added_parameters=0,
        reason='Reuse clear sampling features; replace arbitrary low-range basis with ordered cumulative luminance and chroma'))
    print('CLEAR_REGIONS_INITIALIZED',max(errors),ranges,flush=True)


if __name__=='__main__':main()
