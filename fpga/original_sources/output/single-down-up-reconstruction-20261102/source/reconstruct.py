"""Isolated architecture experiment; no production contract or pack is edited."""
from pathlib import Path
import os, sys, json, time, copy, hashlib, shutil, math, argparse
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
ROOT=Path(r'E:\fpga')
D=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'output/b1-mc-pixel-20261027/source'))
from core import torch, nn, F, HalfAway, setup, save_json, sha
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def graph(single):
    rows=[];prev='input';cin=3
    def conv(name,cout,k=1,stride=1,dw=False,relu=True):
        nonlocal prev,cin
        rows.append(dict(id=name,kind='dw3x3' if dw else ('conv3x3' if k==3 else 'pw1x1'),inputs=[prev],cin=cin,channels=cout,kernel=k,stride=stride,groups=cin if dw else 1,activation='relu' if relu else 'none'))
        prev=name;cin=cout
    def ds(name,cout,stride=1,linear=False):
        conv(name+'_dw',cin,3,stride,True);conv(name+'_pw',cout,relu=not linear)
    conv('stem',8,3)
    if single:ds('down',32,4)
    else:ds('down1',16,2);ds('down2',32,2)
    for i in range(2 if single else 3):
        skip=prev;ds(f'res{i+1}_a',32);ds(f'res{i+1}_b',32,linear=True)
        name=f'res{i+1}_add';rows.append(dict(id=name,kind='add',inputs=[prev,skip],channels=32));prev=name
    for i,(cout,factor) in enumerate([(8,4)] if single else [(16,2),(8,2)]):
        ds(f'up{i+1}_low',cout)
        name=f'up{i+1}_nearest';rows.append(dict(id=name,kind='nearest',inputs=[prev],channels=cout,factor=factor));prev=name
        conv(f'up{i+1}_high',cout,3,1,True)
    conv('rgb',3,relu=False)
    return rows

class Model(nn.Module):
    def __init__(self,single):
        super().__init__();self.rows=graph(single);self.single=single;self.qat=False
        self.w=nn.ParameterDict();self.b=nn.ParameterDict()
        for r in self.rows:
            if 'kernel' not in r:continue
            name=r['id'];shape=(r['channels'],r['cin']//r['groups'],r['kernel'],r['kernel'])
            w=torch.zeros(shape);b=torch.zeros(r['channels'])
            if name=='stem':
                for c in range(8):w[c,(c//2)%3,1,1]=1 if c%2==0 else -1
            elif r['kind']=='dw3x3':
                if name.endswith('_high') or name.startswith('down'):
                    w[:,0]=torch.tensor([[1,2,1],[2,4,2],[1,2,1]])/16
                else:w[:,0,1,1]=1
            elif name=='rgb':
                for c in range(3):w[c,2*c,0,0]=1;w[c,2*c+1,0,0]=-1
            elif name.endswith('_b_pw'):w.normal_(0,.001)
            else:
                for c in range(r['channels']):w[c,c%min(r['cin'],6),0,0]=1
            self.w[name]=nn.Parameter(w);self.b[name]=nn.Parameter(b)
            self.register_buffer(name+'_m',torch.ones(r['channels'],dtype=torch.float64)*1024)
    @torch.no_grad()
    def calibrate(self):
        for name,w in self.w.items():
            m=(w.abs().flatten(1).amax(1)*131072/120).ceil().clamp(64,131071)
            getattr(self,name+'_m').copy_(m.double())
    def forward(self,x,all_nodes=False):
        values={'input':x*128}
        if self.qat:values['input']=HalfAway.apply(values['input']).clamp(-128,127)
        for r in self.rows:
            name=r['id'];a=values[r['inputs'][0]]
            if name in self.w:
                if r['kernel']==3:a=F.pad(a,(1,1,1,1),mode='replicate')
                if self.qat:
                    m=getattr(self,name+'_m');s=(m/131072).float()
                    wi=HalfAway.apply(self.w[name]/s[:,None,None,None]).clamp(-127,127)
                    bi=HalfAway.apply(self.b[name]/s).double()[None,:,None,None]
                    dot=F.conv2d(a,wi,stride=r['stride'],groups=r['groups'])
                    y=HalfAway.apply((dot.double()+bi).clamp(-(2**31),2**31-1)*m[None,:,None,None]/131072).float().clamp(-128,127)
                else:y=F.conv2d(a,self.w[name],self.b[name],stride=r['stride'],groups=r['groups']).clamp(-128,127)
                if r['activation']=='relu':y=y.clamp_min(0)
            elif r['kind']=='nearest':y=a.repeat_interleave(r['factor'],-2).repeat_interleave(r['factor'],-1)
            else:y=(a+values[r['inputs'][1]]).clamp(-128,127)
            values[name]=y
        return values if all_nodes else values['rgb']/128
    @torch.no_grad()
    def export(self,folder):
        arrays={}
        for name,w in self.w.items():
            m=getattr(self,name+'_m');s=(m/131072).float()
            arrays[name+'_weight']=HalfAway.apply(w/s[:,None,None,None]).clamp(-127,127).to(torch.int8).cpu().numpy()
            arrays[name+'_bias']=HalfAway.apply(self.b[name]/s).int().cpu().numpy()
            arrays[name+'_multiplier']=m.int().cpu().numpy()
        np.savez(folder/'integer-coefficients.npz',**arrays)
        save_json(folder/'graph.json',dict(nodes=self.rows,numeric=dict(input='uint8 - 128',activation='int8 scale 1/128',weights='int8 per output channel',bias='int32',requantization='Q17 nearest half away from zero',padding='replicate',residual='clamp_i8(main + skip)',output='int8 + 128'),hardware_compatible=False))

class IntegerModel:
    def __init__(self,folder):
        self.rows=json.loads((folder/'graph.json').read_text('utf-8'))['nodes'];self.ops={}
        with np.load(folder/'integer-coefficients.npz',allow_pickle=False) as z:
            for r in self.rows:
                n=r['id']
                if n+'_weight' not in z:continue
                self.ops[n]=(torch.tensor(z[n+'_weight'].astype(np.float32),device='cuda'),torch.tensor(z[n+'_bias'],dtype=torch.float64,device='cuda')[None,:,None,None],torch.tensor(z[n+'_multiplier'],dtype=torch.float64,device='cuda')[None,:,None,None])
    @torch.inference_mode()
    def __call__(self,x,all_nodes=False):
        v={'input':x*128};assert torch.equal(v['input'],v['input'].round())
        for r in self.rows:
            n=r['id'];a=v[r['inputs'][0]]
            if n in self.ops:
                w,b,m=self.ops[n]
                if r['kernel']==3:a=F.pad(a,(1,1,1,1),mode='replicate')
                dot=F.conv2d(a,w,stride=r['stride'],groups=r['groups'])
                y=HalfAway.apply((dot.double()+b).clamp(-(2**31),2**31-1)*m/131072).clamp(-128,127).float()
                if r['activation']=='relu':y=y.clamp_min(0)
            elif r['kind']=='nearest':y=a.repeat_interleave(r['factor'],-2).repeat_interleave(r['factor'],-1)
            else:y=(a+v[r['inputs'][1]]).clamp(-128,127)
            v[n]=y
        return v if all_nodes else v['rgb']/128

def tx(a):return torch.from_numpy(np.array(a,copy=True)).permute(0,3,1,2).cuda().float()/128-1
def rgb(t):return (t[0].permute(1,2,0).cpu().numpy()*128+128).round().clip(0,255).astype(np.uint8)
def metrics(a,b):
    e=a.astype(np.float64)-b.astype(np.float64);mse=float((e*e).mean())
    return dict(mae=float(np.abs(e).mean()),psnr_db=10*math.log10(255**2/max(mse,1e-12)))
def costs(rows):
    shapes={'input':(720,1280,3)};mac=weights=bias=0
    for r in rows:
        h,w,c=shapes[r['inputs'][0]]
        if 'kernel' in r:
            h=(h+r['stride']-1)//r['stride'];w=(w+r['stride']-1)//r['stride']
            n=r['channels']*(r['cin']//r['groups'])*r['kernel']**2
            weights+=n;bias+=r['channels'];mac+=h*w*n
        elif r['kind']=='nearest':h*=r['factor'];w*=r['factor']
        shapes[r['id']]=(h,w,r['channels'])
    return dict(weights=weights,biases=bias,parameters=weights+bias,macs_per_720p=mac,convolutions=sum('kernel' in r for r in rows),residual_blocks=sum(r['kind']=='add' for r in rows),shapes=shapes)

def data():
    split=json.loads((ROOT/'output/b1-mc-pixel-20261027/data_split.json').read_text('utf-8'))
    manifest=json.loads((ROOT/'output/b1-color-cycle-negative-20261027/acceptance-four-scenes-5s-20261029/source_manifest.json').read_text('utf-8'))
    assert not {x['video'] for x in split['training']}&{x['video'] for x in split['validation']}
    assert not {x['video'] for k in ['training','validation'] for x in split[k]}&{Path(x['source']).name for x in manifest['scenes']}
    rng=np.random.default_rng(20261102);parts=[];records=[]
    for key in ['training','validation']:
        out=[]
        for r in split[key]:
            assert sha(r['path'])==r['sha256']
            a=np.asarray(Image.open(r['path']).convert('RGB'));h,w=a.shape[:2]
            for j in range(8 if key=='training' else 8):
                yy=int(rng.integers(0,(h-160)//4+1))*4;xx=int(rng.integers(0,(w-160)//4+1))*4
                out.append(a[yy:yy+160,xx:xx+160].copy());records.append(dict(split=key,path=r['path'],sha256=r['sha256'],y=yy,x=xx,size=160))
        parts.append(tx(np.stack(out)))
    save_json(D/'data.json',dict(source_split=str(ROOT/'output/b1-mc-pixel-20261027/data_split.json'),source_disjoint=True,crops=records,counts=[len(x) for x in parts],seed=20261102))
    return parts

def fit(single,X,V,steps_fp=1000,steps_qat=300):
    key='single4x-r2' if single else 'original2x2x-r3';folder=D/key;folder.mkdir(exist_ok=False)
    setup(20261102);m=Model(single).cuda();rng=np.random.default_rng(20261102);logs=[];start=time.perf_counter()
    def score():
        with torch.no_grad():return float(torch.cat([m(z).detach() for z in V.split(4)]).sub(V).abs().mean()*128)
    initial=score()
    for stage,steps,lr in [('fp32',steps_fp,.0015),('qat',steps_qat,.00025)]:
        m.qat=stage=='qat'
        if m.qat:m.calibrate()
        opt=torch.optim.Adam(m.parameters(),lr=lr);best=score();state=copy.deepcopy(m.state_dict());selected=0
        for step in range(steps+1):
            if time.perf_counter()-start>600:raise TimeoutError('10 minute training cap per model')
            if step%100==0 or step==steps:
                val=score();logs.append(dict(stage=stage,step=step,validation_mae=val,seconds=time.perf_counter()-start))
                if val<=best:best=val;state=copy.deepcopy(m.state_dict());selected=step
                print(key,stage,step,'MAE',round(val,3),'best',selected,'sec',round(time.perf_counter()-start,1),flush=True)
            if step==steps:break
            ids=torch.as_tensor(rng.choice(len(X),8,replace=False),device='cuda');x=X[ids]
            # Whole-batch flips preserve 4-pixel phase alignment and source separation.
            if step%2:x=x.flip(-1)
            opt.zero_grad(set_to_none=True);y=m(x)
            loss=(y-x).abs().mean()+.25*(y-x).square().mean()
            loss+=.12*((y[:,:,:,1:]-y[:,:,:,:-1])-(x[:,:,:,1:]-x[:,:,:,:-1])).abs().mean()
            loss+=.12*((y[:,:,1:]-y[:,:,:-1])-(x[:,:,1:]-x[:,:,:-1])).abs().mean()
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),3);opt.step()
        m.load_state_dict(state)
        torch.save(dict(state_dict=m.state_dict(),single=single,qat=m.qat,selected_step=selected,gradient_steps=steps),folder/(stage+'.pt'))
    m.eval();m.export(folder);engine=IntegerModel(folder)
    with torch.no_grad():
        for x in V.split(4):assert torch.equal(engine(x),m(x))
    result=dict(model=key,initial_validation_mae=initial,selected_validation_mae=score(),fp32_steps=steps_fp,qat_steps=steps_qat,records=logs,cost=costs(m.rows),export_sha256=sha(folder/'integer-coefficients.npz'),qat_vs_export_validation_exact=True,no_input_output_bypass=True,normalization='none',training_seconds=time.perf_counter()-start)
    save_json(folder/'training.json',result)
    return result

def oracle(folder):
    from network_ab.contract_reference import convolution
    z=np.load(folder/'integer-coefficients.npz',allow_pickle=False);engine=IntegerModel(folder);rng=np.random.default_rng(717)
    compared=0
    for a in [rng.integers(0,256,(8,12,3),dtype=np.uint8),np.full((8,8,3),255,np.uint8)]:
        values={'input':(a.astype(int)-128).tolist()};actual=engine(tx(a[None]),True)
        for r in engine.rows:
            name=r['id'];x=values[r['inputs'][0]]
            if 'kernel' in r:
                w=z[name+'_weight'].transpose(0,2,3,1).reshape(r['channels'],-1).astype(int).tolist()
                y,_=convolution(x,r,w,z[name+'_bias'].astype(int).tolist(),z[name+'_multiplier'].astype(int).tolist())
            elif r['kind']=='nearest':
                f=r['factor'];y=[[list(x[i//f][j//f]) for j in range(len(x[0])*f)] for i in range(len(x)*f)]
            else:y=np.clip(np.asarray(x)+np.asarray(values[r['inputs'][1]]),-128,127).tolist()
            values[name]=y;expected=np.asarray(y);got=actual[name][0].permute(1,2,0).cpu().numpy().astype(int)
            assert np.array_equal(expected,got),(name,np.abs(expected-got).max());compared+=expected.size
    z.close();return dict(all_nodes_exact=True,scalar_comparisons=compared,oracle='existing independent Python integer convolution, generic nearest4 and clamp-sum')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['train','review']);args=parser.parse_args();setup(20261102)
    assert shutil.disk_usage(D).free>8*2**30
    if args.mode=='train':
        X,V=data();reports=[]
        for single in [True,False]:
            r=fit(single,X,V);r['oracle']=oracle(D/r['model']);reports.append(r)
            save_json(D/'training-summary.json',dict(models=reports,complete=len(reports)==2,hardware_modified=False))
    else:
        from review import run
        run()
if __name__=='__main__':main()
