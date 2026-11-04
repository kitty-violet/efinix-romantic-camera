"""CPU-only initialization using training frames, never validation fitting."""
import copy,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from .ordinal_coarse_net import OrdinalCoarseNet
from .coarse_skip_net import CoarseSkipNet
from .train_luma_regions import ROOT,RUN,components,centers,display,SLICES
from .train_b1_distillation import as_tensor,read_image,to_rgb
from .runtime import write_json,file_hash


@torch.no_grad()
def main():
    torch.set_num_threads(2);root=ROOT/'runs'/RUN;initial=root/'aligned-regions-fp32/fp32_checkpoint_001500.pt';out=root/'ordinal-regions-fp32'
    assert not out.exists() or not any(out.iterdir()), 'Refuse overwrite of populated run'
    out.mkdir(exist_ok=True)
    state=torch.load(initial,map_location='cpu',weights_only=True);assert all(file_hash(k)==v for k,v in state['binding']['code'].items())
    source=CoarseSkipNet().eval();source.load_state_dict(state['model']);data=json.loads((initial.parent/'data.json').read_text(encoding='utf-8'))
    xx=np.zeros((9,9),np.float64);xy=np.zeros((9,3),np.float64)
    for i,r in enumerate(data['pairs']):
        assert file_hash(r['current'])==r['current_sha256'] and file_hash(r['current_target'])==r['current_target_sha256']
        v=source(as_tensor(read_image(r['current']))[None],True)['finish.dw'][0,:,::8,::8].reshape(8,-1).T.numpy().astype(np.float64)
        f=np.concatenate((v,np.ones((len(v),1))),1);t=components(as_tensor(read_image(r['current_target']))[None])[0,:,::8,::8].reshape(3,-1).T.numpy().astype(np.float64)
        xx+=f.T@f;xy+=f.T@t
        if (i+1)%10==0:print('ORDINAL_INITIAL_FIT',i+1,flush=True)
    coef=np.linalg.lstsq(xx+np.eye(9)*1e-6,xy,rcond=1e-12)[0];net=OrdinalCoarseNet(state['model'],torch.from_numpy(coef).float()).eval();folded=net.folded()
    x=torch.rand(1,3,32,48)*2-1;a=net(x,True);b=folded(x);err=float((a['rgb']-b).abs().max());assert err==0
    z=net.component_prediction(a['finish.dw']);code=[(z[:,i,None]-c[None,:,None,None]).abs().argmin(1) for i,c in enumerate(centers(x))]
    mismatches=sum(int((a['rgb'][:,s].argmax(1)!=q).sum()) for s,q in zip(SLICES,code));assert mismatches==0
    torch.save(dict(model=net.state_dict(),scales=None,mode='ordinal_initialization',source=str(initial),source_sha256=file_hash(initial)),out/'initial.pt')
    for k in ['data.json','graph.json']:(out/k).write_bytes((initial.parent/k).read_bytes())
    cfg=copy.deepcopy(state['config']);cfg.update(seed=2026100446,initial=str(out/'initial.pt'),initial_sha256=file_hash(out/'initial.pt'),fp32_steps=2000,fp32_lr=.0005,
        objective_weights=dict(ce=.25,color=2,flat=32,region=2,temporal=3,phase=3),ordered_head_temperature=[128.,256.,512.],
        initialization='Least squares fit three continuous target components on 95 training frames; categorical head mathematically tied to ordered levels; same exported C32 PW')
    write_json(out/'config.json',cfg);binding=copy.deepcopy(state['binding'])
    for f in [Path(__file__),Path(__file__).with_name('ordinal_coarse_net.py')]:binding['code'][str(f)]=file_hash(f)
    binding.update(config_sha256=file_hash(out/'config.json'),data_sha256=file_hash(out/'data.json'));write_json(out/'binding.json',binding)
    folder=out/'initial_images';folder.mkdir()
    for r in data['validation']:Image.fromarray(to_rgb(display(net(as_tensor(read_image(r['source']))[None]))[0])).save(folder/(r['id']+'.png'))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16);board=Image.new('RGB',(960,894),'#18202a');draw=ImageDraw.Draw(board)
    for j,(key,(x,y)) in enumerate([('val_stairs_090',(150,430)),('val_stairs_090',(424,320)),('val_plaza_090',(80,400))]):
        for i,(path,title) in enumerate([(initial.parent/'fp32_001500'/(key+'.png'),'原自由分类头'),(root/'intermediate-target-screen'/f'{key}_0.008.png','区域目标'),(folder/(key+'.png'),'有序色阶头初始化')]):
            draw.text((320*i+5,298*j+5),title,font=font,fill='white');board.paste(Image.open(path).crop((x,y,x+320,y+264)),(320*i,298*j+30))
    board.save(folder/'comparison.png');write_json(out/'initialization.json',dict(training_frames=len(data['pairs']),projection=coef.tolist(),fold_max_error=err,nearest_level_mismatches=mismatches,graph_unchanged=True,style_accepted=False))
    print('ORDINAL_INITIALIZED',err,mismatches,flush=True)


if __name__=='__main__':main()
