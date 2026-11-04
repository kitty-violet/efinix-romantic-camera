"""Predict one shared surface value, then use fixed ordered colour levels.

All fourteen threshold rows have identical learned inputs. Thresholds and
positive spatial reconstruction are compiled into the existing B1 tail.
This experiment tests shared level geometry, not independent bit decisions.
"""
import argparse,copy,json,shutil,time
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .probe_b1_encoder_gauge import load
from .audit_b1_structural_task_updates import restored_optimizer
from .probe_b1_decoder_coverage import fragment
from .train_b1_trainable_luma_readout import ReadoutB1
from .train_b1_winner_surrogate import ExactValueProxyGradient
from .runtime import configure_workspace,write_json,file_hash
from .engine import set_reproducible
from .train_b1_late_region_supervision import executed_sources
from . import train_b1_minimum_area as m


def surface(rgb):
    r,g,b=(rgb*128+128).split(1,1)
    return torch.cat(((r+2*g+b)/8,(r-b)/4+64,(2*g-r-b)/8+64),1)


@torch.no_grad()
def install(net):
    keys=('up1_high_dw','up2_low_dw','up2_low_pw','up2_high_dw','rgb_head')
    for k in keys:net.weight[k].zero_();net.bias[k].zero_()
    kernel=net.weight['stem'].new_tensor([[1,2,1],[2,4,2],[1,2,1]])/16
    net.weight['up1_high_dw'][:,0]=kernel
    net.up1_high_dw_m.fill_(1024)
    # Values are RGB/2, with levels8+17*n in ordinary8bit luminance.
    for j in range(14):
        threshold=(8+17*(j+.5))/2
        gain=127*(131071/131072)
        net.weight['up2_low_dw'][j,0,1,1]=gain
        net.bias['up2_low_dw'][j]=-gain*int(threshold)
    net.up2_low_dw_m.fill_(65536)
    net.up2_low_dw_m[:,:14]=131071
    for j in (14,15):net.weight['up2_low_dw'][j,0,1,1]=1
    net.weight['up2_low_pw'][0,:14,0,0]=8257/131072
    net.up2_low_pw_m.fill_(2048);net.up2_low_pw_m[:,0]=8257
    for j in (1,2):net.weight['up2_low_pw'][j,j+13,0,0]=1
    # Only one positive3x3 reconstruction after the360p decisions.
    for j in range(3):net.weight['up2_high_dw'][j,0]=kernel
    net.up2_high_dw_m.fill_(1024)
    net.weight['rgb_head'][:,:3,0,0]=torch.tensor([[2.125,2,-2],[2.125,0,2],[2.125,-2,-2]],device=kernel.device)
    net.bias['rgb_head'].copy_(torch.tensor([-120.,-248.,136.],device=kernel.device))
    net.rgb_head_m.fill_(4096)
    for k in keys:
        s=(getattr(net,k+'_m')/131072).float().flatten()[:,None,None,None]
        assert (net.weight[k]/s).abs().max()<=127,k


class SharedSurfaceB1(ReadoutB1):
    def training_activation_proxy(self,name,value,pre,row):
        if name=='up2_low_dw':
            self.surface360=pre
            # The threshold's surrogate derivative is1 in surface units.
            # Actual inference still uses the exact signed-byte clipping.
            # Chroma rows are ordinary identity maps, so keep their gradient.
            return ExactValueProxyGradient.apply(value,torch.cat((pre[:,:14]/127,pre[:,14:]),1))
        return super().training_activation_proxy(name,value,pre,row)


@torch.no_grad()
def check_decoder(net):
    z=torch.zeros(256,16,1,1,device=next(net.parameters()).device)
    z[:,:14]=torch.arange(256,device=z.device)[:,None,None,None]/2
    z[:,14:]=64
    p=fragment(net,net.round(z),'up1_low_pw')['rgb_head'][:,:,0,0]+128
    assert (torch.diff(p[:,0])>=0).all() and (p[:,0]==p[:,1]).all() and (p[:,0]==p[:,2]).all()
    assert len(p[:,0].unique())==15
    return dict(grey_levels=15,neutral_axis_exact=True,monotone=True)


def main():
    ap=argparse.ArgumentParser()
    for k in ('checkpoint','data','mining','out'):ap.add_argument('--'+k,required=True)
    ap.add_argument('--steps',type=int,default=3600);ap.add_argument('--local',action='store_true')
    a=ap.parse_args();configure_workspace();set_reproducible(81025);torch.set_num_threads(2)
    out=Path(a.out);out.mkdir(exist_ok=False);state,net=load(a.checkpoint);net.cuda();net.__class__=SharedSurfaceB1
    data=m.load_data(a.data);rows={r['id']:r for r in data['training']}
    rr=json.loads(Path(a.mining).read_text('utf-8'))['records'];specs=[rr[i] for i in (1,5,6)]
    to_tensor=lambda ar:torch.from_numpy(np.stack(ar).transpose(0,3,1,2).copy()).float().cuda()/128-1
    def crops(specs):
        xx=[];tt=[]
        for s in specs:
            row=rows[s['row']['id']];x,y=s['xy'];w,h=s['wh']
            xx.append(m.t.r.read(row['path'])[y:y+h,x:x+w]);tt.append(m.t.r.read(row['target'])[y:y+h,x:x+w])
        return to_tensor(xx),to_tensor(tt)
    fixedx,fixedt=crops(specs);cut=lambda z:z[...,40:-40,40:-40]
    with torch.no_grad():old=net(fixedx)[0]
    # TRAIN-only linear initialization of the common surface and two chroma.
    calibration=[]
    for video in sorted({r['video'] for r in rows.values()}):
        group=[r for r in rows.values() if r['video']==video]
        calibration.extend(group[i] for i in sorted(set((0,len(group)//2,len(group)-1))))
    xx=[];yy=[]
    with torch.no_grad():
        for row in calibration:
            x=to_tensor([m.t.r.read(row['path'])]);t=to_tensor([m.t.r.read(row['target'])])
            f=net(x,all_nodes=True)['up1_low_dw']
            xx.append(f[0,:,::3,::3].flatten(1).T);yy.append(F.avg_pool2d(surface(t),4)[0,:,::3,::3].flatten(1).T)
        x=torch.cat(xx).double();y=torch.cat(yy).double();mu=x.mean(0);sd=x.std(0).clamp_min(2)
        v=torch.cat(((x-mu)/sd,torch.ones(len(x),1,device=x.device)),1)
        ridge=torch.eye(33,device=x.device,dtype=torch.float64)*len(x)*.001;ridge[-1,-1]=0
        theta=torch.linalg.solve(v.T@v+ridge,v.T@y)
        w=(theta[:32].T/sd).float();b=(theta[-1]-mu@(theta[:32]/sd[:,None])).float()
        k='up1_low_pw';net.weight[k][:14,:,0,0]=w[0];net.weight[k][14:,:,0,0]=w[1:]
        net.bias[k][:14]=b[0];net.bias[k][14:]=b[1:]
        quantum=(net.weight[k].abs().flatten(1).amax(1)/120*131072).ceil().clamp(1,131071)
        net.up1_low_pw_m.copy_(quantum.double()[None,:,None,None]);install(net)
    checks=check_decoder(net);fixed=False
    for k,w in net.weight.items():
        if k=='up1_high_dw':fixed=True
        w.requires_grad_(not fixed);net.bias[k].requires_grad_(not fixed)
    opt=restored_optimizer(net,state['optimizer']);opt.state.clear()
    for g in opt.param_groups:
        for p in g['params']:
            if p is net.weight['up1_low_pw']:g['row_gain']=(net.up1_low_pw_m/512).float().flatten()[:,None,None,None]
            if p is net.bias['up1_low_pw']:g['lr']=.02
    frozen={k:(w.detach().clone(),net.bias[k].detach().clone()) for k,w in net.weight.items() if not w.requires_grad}
    def update(x,t):
        opt.zero_grad(set_to_none=True);nodes=net(x,all_nodes=True);p=nodes['rgb_head']/128
        fields=torch.cat((nodes['up1_low_pw'][:,:1],nodes['up1_low_pw'][:,14:]),1)
        want=F.avg_pool2d(surface(t),4)
        low=(fields[...,10:-10,10:-10]-want[...,10:-10,10:-10]).abs().mean()/32
        colour=(cut(p)-cut(t)).abs().mean()
        # Finer output matching prevents a good180p fit hiding decoder errors.
        loss=low+2*colour;loss.backward()
        for p in (net.weight['up1_low_pw'],net.bias['up1_low_pw']):p.grad[:14]=p.grad[:14].sum(0,keepdim=True)
        opt.prepare_gradients();gn=torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True)
        assert gn>0 and torch.isfinite(loss);opt.step()
        assert torch.equal(net.weight['up1_low_pw'][:14],net.weight['up1_low_pw'][:1].expand_as(net.weight['up1_low_pw'][:14]))
        assert torch.equal(net.bias['up1_low_pw'][:14],net.bias['up1_low_pw'][:1].expand(14))
        return float(loss.detach()),float(low.detach()),float(colour.detach())
    initial=copy.deepcopy(net.state_dict());initialopt=copy.deepcopy(opt.state_dict())
    update(fixedx,fixedt);torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict()),out/'restore_probe.pt')
    expected=update(fixedx,fixedt);snap=copy.deepcopy(net.state_dict());disk=torch.load(out/'restore_probe.pt',map_location='cuda',weights_only=True)
    net.load_state_dict(disk['model']);opt.load_state_dict(disk['optimizer']);observed=update(fixedx,fixedt)
    err=max(float((v-net.state_dict()[k]).abs().max()) for k,v in snap.items());assert err==0 and expected==observed
    net.load_state_dict(initial);opt.load_state_dict(initialopt)
    paths=sorted(set(executed_sources(type(net))[0]+[Path(__file__)]));hashes={str(p):file_hash(p) for p in paths}
    for p in paths:shutil.copy2(p,out/p.name)
    write_json(out/'preflight.json',dict(regression=checks,gpu_forward_backward_minibatch=True,disk_next_step_error=err))
    write_json(out/'config.json',dict(args=vars(a),code=hashes,data_sha256=file_hash(a.data),calibration_ids=[r['id'] for r in calibration],inference_additions=0))
    print('SHARED_SCALAR_PREFLIGHT GPU/backward/minibatch/disk0/15monotonelevels',flush=True)
    history=[];font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',12);start=time.monotonic()
    def review(step):
        for k,(w,b) in frozen.items():assert torch.equal(net.weight[k],w) and torch.equal(net.bias[k],b)
        with torch.no_grad():
            raw=net.export();assert len(raw)==20480;nodes=net(fixedx,all_nodes=True)
            actual,_=m.t.r.IntegerB1(m.t.r.runtime_from_image(raw),device='cuda')(net.round(fixedx*128),True)
            assert all(torch.equal(v,actual[k]) for k,v in nodes.items());p=actual['rgb_head']/128
        board=Image.new('RGB',(960,792),'#17202a');draw=ImageDraw.Draw(board)
        for i in range(3):
            for j,(title,z) in enumerate((('原片',fixedx),('固定目标',fixedt),('原B1',old),('共同色层·实际B1',p))):
                board.paste(m.t.r.c.rgb_image(cut(z)[i]),(240*j,264*i+24));draw.text((240*j+3,264*i+3),title,font=font,fill='white')
        board.save(out/f'comparison_{step:06d}.png');(out/f'coefficients_{step:06d}.bin').write_bytes(raw)
        torch.save(dict(model=net.state_dict(),optimizer=opt.state_dict(),step=step,contract_sha256=state['contract_sha256'],residual_encoding=net.package['residual_encoding'],qat=True,shared_scalar_surface=True,visual_acceptance=False),out/f'candidate_{step:06d}.pt')
        row=dict(step=step,local_diagnostic_mae=float(cut(p-fixedt).abs().mean()*128),all_nodes_error=0)
        history.append(row);write_json(out/'history.json',history);print('SHARED_SCALAR_REVIEW',row,flush=True)
    review(0);gen=np.random.default_rng(81025);allrows=list(rows.values());cache={}
    def batch():
        if a.local:return fixedx,fixedt
        xx=[];tt=[]
        for index in gen.integers(0,len(allrows),3):
            row=allrows[index]
            if row['id'] not in cache:cache[row['id']]=(m.t.r.read(row['path']),m.t.r.read(row['target']))
            src,tgt=cache[row['id']];y=int(gen.integers(0,101))*4;x=int(gen.integers(0,241))*4
            flip=bool(gen.integers(2));src=src[y:y+320,x:x+320];tgt=tgt[y:y+320,x:x+320]
            if flip:src=src[:,::-1];tgt=tgt[:,::-1]
            xx.append(src);tt.append(tgt)
        return to_tensor(xx),to_tensor(tt)
    for step in range(a.steps):
        values=update(*batch())
        if (step+1)%100==0:print('SHARED_SCALAR_STEP',step+1,values,flush=True)
        if (step+1)%400==0 or step+1==a.steps:
            assert all(file_hash(k)==v for k,v in hashes.items());review(step+1)
            if (out/'STOP_REQUEST').exists():break
    write_json(out/'completion.json',dict(history=history,seconds=time.monotonic()-start,visual_acceptance=False,video_not_validated=True))


if __name__=='__main__':main()
