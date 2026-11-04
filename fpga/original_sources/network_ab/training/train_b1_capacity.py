"""Matched 20-KiB-budget capacity experiment, including a continued small control.

Both networks see exactly the same native/cropped adjacent samples, noise,
updates, objectives and schedule. No new data or spatial postfilter is added.
"""
import argparse,copy,gc,json,time
from pathlib import Path
import torch
from PIL import Image,ImageDraw,ImageFont
from .train_detail_rgb import Training as DataTraining
from .train_component_centers import Training as CenterTraining,spatial,target_components
from .component_region_net import ComponentNet,decode
from .b1_capacity_component_net import CapacityComponentNet,plan,budget
from .train_luma_regions import ROOT,RUN
from .train_b1_distillation import as_tensor,read_image,to_rgb
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible


class Training(CenterTraining):
    def __init__(self,out):
        DataTraining.__init__(self,out,'fp32');p=Path(self.cfg['initial']);assert file_hash(p)==self.cfg['initial_sha256']
        cls=CapacityComponentNet if self.cfg['variant']=='capacity' else ComponentNet
        self.net=cls().cuda().eval();self.net.load_state_dict(torch.load(p,map_location='cpu',weights_only=True)['model'])
        self.opt=torch.optim.Adam(self.net.parameters(),lr=self.cfg['fp32_lr'])

    @torch.no_grad()
    def evaluate(self):
        folder=self.out/f'fp32_{self.step:06d}';folder.mkdir(exist_ok=False);rows=[]
        for r in self.data['validation']:
            x=as_tensor(read_image(r['source']))[None].cuda();raw=self.net(x);pred=decode(raw)
            target=as_tensor(read_image(ROOT/'runs'/RUN/'intermediate-target-screen'/f'{r["id"]}_0.008.png'))[None].cuda()
            parts=spatial(raw,target,24)
            Image.fromarray(to_rgb(pred[0])).save(folder/(r['id']+'.png'))
            rows.append(dict(id=r['id'],**{k:float(v)*128 for k,v in parts.items()},rgb_error=float((pred-decode(target_components(target))).abs().mean()*128)))
        write_json(folder/'scores.json',dict(records=rows,not_blind=True,style_accepted=False,variant=self.cfg['variant']))


def prepare(root,steps):
    source=ROOT/'runs'/RUN/'component-centers-fp32/fp32_checkpoint_003000.pt';s=torch.load(source,map_location='cpu',weights_only=True)
    assert s['scales'] is None and all(file_hash(k)==v for k,v in s['binding']['code'].items())
    root.mkdir(exist_ok=False);small=ComponentNet().eval();small.load_state_dict(s['model']);large=CapacityComponentNet().eval();large.widen_from(small)
    b=budget();assert b['trainable_parameters']<=11379;write_json(root/'budget.json',b)
    small=small.cuda();large=large.cuda();checks=[]
    with torch.no_grad():
        for shape in [(2,3,96,160),(1,3,720,1280)]:
            x=torch.rand(shape,device='cuda')*1.9921875-1;a=small(x);p=large(x);err=float((a-p).abs().max());assert err<2e-5
            checks.append(dict(shape=shape,max_abs_error=err))
    write_json(root/'initial_function_checks.json',dict(checks=checks,exact_algebraic_widening=True,fp32_roundoff_tolerance=2e-5,zero_sum_channel_noise=True))
    for variant,net in [('capacity',large),('small-control',small)]:
        out=root/variant;out.mkdir();initial=out/'initial.pt';torch.save(dict(model={k:v.cpu() for k,v in net.state_dict().items()},source=str(source),source_sha256=file_hash(source)),initial)
        (out/'data.json').write_bytes((source.parent/'data.json').read_bytes());write_json(out/'graph.json',plan() if variant=='capacity' else json.loads((source.parent/'graph.json').read_text()))
        cfg=copy.deepcopy(s['config']);cfg.update(seed=2026100455,variant=variant,initial=str(initial),initial_sha256=file_hash(initial),fp32_steps=steps,fp32_lr=.0004,precision='FP32 no AMP/TF32/EMA',matched_capacity_control=True)
        write_json(out/'config.json',cfg);binding=copy.deepcopy(s['binding'])
        for f in [Path(__file__),Path(__file__).with_name('b1_capacity_component_net.py')]:binding['code'][str(f)]=file_hash(f)
        binding.update(config_sha256=file_hash(out/'config.json'),data_sha256=file_hash(out/'data.json'));write_json(out/'binding.json',binding)
    print('CAPACITY_PREPARED',b['raw_weights'],b['trainable_parameters'],b['ideal_macs'],flush=True)


def preflight(root):
    results=[]
    for variant in ['capacity','small-control']:
        out=root/variant;tr=Training(out);calls=[];hooks=[m.register_forward_hook(lambda m,a,y:calls.append(y.numel()*m.weight[0].numel())) for m in tr.net.modules() if isinstance(m,torch.nn.Conv2d)]
        with torch.no_grad():y=tr.net(torch.zeros(1,3,720,1280,device='cuda'));assert y.shape==(1,3,720,1280)
        for h in hooks:h.remove()
        assert sum(calls)==json.loads((out/'graph.json').read_text())['macs']
        tr.update(True);gc.collect();torch.cuda.empty_cache();tr.update(False);torch.save(tr.state(),out/'preflight.pt');a=tr.update(False)
        expected={k:v.cpu().clone() for k,v in tr.net.state_dict().items()};del tr;gc.collect();torch.cuda.empty_cache()
        tr=Training(out);tr.restore(torch.load(out/'preflight.pt',map_location='cpu',weights_only=True));c=tr.update(False)
        err=max(float((expected[k]-v.cpu()).abs().max()) for k,v in tr.net.state_dict().items());assert err<2e-6 and a['sample']==c['sample'] and a['noise']==c['noise']
        result=dict(variant=variant,status='PASS',actual720_pair_backward=True,batch2=True,restore_max_error=err,macs=sum(calls));results.append(result);write_json(out/'preflight.json',result)
        print('CAPACITY_PREFLIGHT',result,flush=True);del tr;gc.collect();torch.cuda.empty_cache()
    write_json(root/'preflight.json',dict(status='PASS',results=results))


def comparison(root,step):
    data=json.loads((root/'capacity/data.json').read_text(encoding='utf-8'));board=Image.new('RGB',(1280,1192),'#18202a');d=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
    for j,(key,(x,y)) in enumerate([('val_stairs_090',(150,430)),('val_stairs_090',(424,320)),('val_plaza_090',(80,400)),('val_new_hallway_120',(700,280))]):
        r=next(r for r in data['validation'] if r['id']==key)
        paths=[(r['source'],'原片'),(ROOT/'runs'/RUN/'intermediate-target-screen'/f'{key}_0.008.png','区域目标'),(root/'small-control'/f'fp32_{step:06d}'/(key+'.png'),f'小网络同样续训 {step}'),(root/'capacity'/f'fp32_{step:06d}'/(key+'.png'),f'B1容量新网络 {step}')]
        for i,(p,title) in enumerate(paths):d.text((i*320+5,j*298+5),title,font=font,fill='white');board.paste(Image.open(p).crop((x,y,x+320,y+264)),(i*320,j*298+30))
    board.save(root/f'comparison_{step:06d}.png')


def train(root,stop,resume):
    assert json.loads((root/'preflight.json').read_text())['status']=='PASS';models=[Training(root/v) for v in ['capacity','small-control']]
    if resume:
        for tr in models:tr.restore(torch.load(tr.out/f'fp32_checkpoint_{resume:06d}.pt',map_location='cpu',weights_only=True))
    else:
        assert not (root/'paired_metrics.jsonl').exists()
        for tr in models:tr.evaluate()
        comparison(root,0)
    # Reset once so construction/evaluation cannot give one model different noise.
    if not resume:
        torch.manual_seed(2026100455);torch.cuda.manual_seed_all(2026100455)
    write_json(root/f'launch_{resume:06d}.json',dict(start=resume,stop=stop,budget=budget(),source_hashes=models[0].binding['code'],same_sample_noise_and_shift=True))
    with (root/'paired_metrics.jsonl').open('a',encoding='utf-8',buffering=1) as log:
        while models[0].step<min(stop,models[0].cfg['fp32_steps']):
            rng=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all();a=models[0].update()
            torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda);b=models[1].update()
            assert a['sample']==b['sample'] and a['noise']==b['noise'] and a['step']==b['step'] and models[0].rng.getstate()==models[1].rng.getstate()
            row=dict(step=a['step'],capacity=a,small_control=b);log.write(json.dumps(row)+'\n')
            if a['native']:gc.collect();torch.cuda.empty_cache()
            if a['step']%50==0:print('CAPACITY_MATCHED',a['step'],round(a['loss'],4),round(b['loss'],4),flush=True)
            if a['step']%500==0 or a['step']==stop:
                for tr in models:torch.save(tr.state(),tr.out/f'fp32_checkpoint_{tr.step:06d}.pt');tr.evaluate()
                comparison(root,a['step']);gc.collect();torch.cuda.empty_cache()
            if (root/'stop_after_update').exists():
                for tr in models:torch.save(tr.state(),tr.out/f'fp32_checkpoint_{tr.step:06d}.pt')
                return
    print('CAPACITY_MATCHED_DONE',models[0].step,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','preflight','train']);p.add_argument('--name',default='b1-capacity-control-v2');p.add_argument('--steps',type=int,default=2000);p.add_argument('--stop',type=int,default=2000);p.add_argument('--resume-step',type=int,default=0);a=p.parse_args()
    configure_workspace();contract_guard();set_reproducible(2026100455);root=ROOT/'runs'/RUN/a.name
    if a.action=='prepare':prepare(root,a.steps)
    elif a.action=='preflight':preflight(root)
    else:train(root,a.stop,a.resume_step)


if __name__=='__main__':main()
