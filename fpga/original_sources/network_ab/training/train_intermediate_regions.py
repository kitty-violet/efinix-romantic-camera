"""Same checked 19-conv graph, less fragmented intermediate region targets."""
import argparse,copy,gc,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from .train_hard_regions import Training as HardTraining
from .train_detail_rgb import Training as DataTraining
from .skip_filter_net import SkipFilterNet
from .screen_intermediate_regions import target
from .train_luma_regions import ROOT,RUN
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible


def prepare(out):
    # CPU-only offline target generation may run alongside the preceding GPU job.
    torch.set_num_threads(2);root=ROOT/'runs'/RUN;base=root/'native-regions-fp32';initial=base/'fp32_checkpoint_000500.pt'
    state=torch.load(initial,map_location='cpu',weights_only=True);assert all(file_hash(k)==v for k,v in state['binding']['code'].items())
    out.mkdir(exist_ok=False);folder=out/'targets';folder.mkdir();data=json.loads((base/'data.json').read_text(encoding='utf-8'));cfg=copy.deepcopy(state['config'])
    cfg.update(seed=2026100438,initial=str(initial),initial_sha256=file_hash(initial),fp32_steps=3000,fp32_lr=.0003,
        target_lambda=.008,target_gamma=.65,target_sigma=.65,target_outline=.4,
        initial_selection='Native 500 is a familiar-regression candidate with less fragmentation; not blind selection')
    with torch.no_grad():
        for i,r in enumerate(data['pairs']):
            for k in ['current','next']:
                assert file_hash(r[k])==r[k+'_sha256'];x=torch.from_numpy(np.asarray(Image.open(r[k]).convert('RGB')).copy()).permute(2,0,1)[None].float()/255
                y=(target(x,.008)[0].permute(1,2,0).numpy()*255).round().clip(0,255).astype(np.uint8);path=folder/f'{i:03d}_{k}.png';Image.fromarray(y).save(path)
                r[k+'_target']=str(path);r[k+'_target_sha256']=file_hash(path)
            if (i+1)%10==0:print('INTERMEDIATE_PREPARE',i+1,len(data['pairs']),flush=True)
    data['target_kind']='intermediate_lambda008';write_json(out/'data.json',data);write_json(out/'config.json',cfg);(out/'graph.json').write_bytes((base/'graph.json').read_bytes())
    binding=copy.deepcopy(state['binding'])
    for p in [Path(__file__),Path(__file__).with_name('screen_intermediate_regions.py')]:binding['code'][str(p)]=file_hash(p)
    binding.update(config_sha256=file_hash(out/'config.json'),data_sha256=file_hash(out/'data.json'));write_json(out/'binding.json',binding)


class Training(HardTraining):
    def __init__(self,out):
        DataTraining.__init__(self,out,'fp32');p=Path(self.cfg['initial']);assert file_hash(p)==self.cfg['initial_sha256']
        s=torch.load(p,map_location='cpu',weights_only=True);self.net=SkipFilterNet().cuda().eval();self.net.load_state_dict(s['model']);self.net.scales=None;self.opt=torch.optim.Adam(self.net.parameters(),lr=self.cfg['fp32_lr'])

    @torch.no_grad()
    def evaluate(self):
        super().evaluate()
        # Replace only the contact-sheet reference; inherited full output/phase
        # checks are unchanged and the old target is not used for acceptance.
        root=ROOT/'runs'/RUN;folder=self.out/f'{self.mode}_{self.step:06d}';font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
        board=Image.new('RGB',(960,894),'#18202a');d=ImageDraw.Draw(board)
        for j,(key,xy) in enumerate([('val_stairs_090',(150,430)),('val_stairs_090',(424,320)),('val_plaza_090',(80,400))]):
            r=next(r for r in self.data['validation'] if r['id']==key);x,y=xy
            for i,(path,title) in enumerate([(r['source'],'原片'),(root/'intermediate-target-screen'/f'{key}_0.008.png','中间区域目标 .008'),(folder/(key+'.png'),f'FP32 {self.step}步')]):
                d.text((320*i+5,298*j+5),title,font=font,fill='white');board.paste(Image.open(path).crop((x,y,x+320,y+264)),(320*i,298*j+30))
        board.save(folder/'comparison.png')


def preflight(out):
    tr=Training(out);tr.update(True);gc.collect();torch.cuda.empty_cache();tr.update(False);torch.save(tr.state(),out/'preflight.pt');a=tr.update(False)
    exp={k:v.cpu().clone() for k,v in tr.net.state_dict().items()};del tr;gc.collect();torch.cuda.empty_cache()
    tr=Training(out);tr.restore(torch.load(out/'preflight.pt',map_location='cpu',weights_only=True));b=tr.update(False)
    err=max(float((exp[k]-v.cpu()).abs().max()) for k,v in tr.net.state_dict().items());assert err<2e-6 and a['sample']==b['sample'] and a['noise']==b['noise']
    write_json(out/'preflight.json',dict(status='PASS',actual720_pair_backward=True,batch2=True,restore_max_error=err));print('INTERMEDIATE_PREFLIGHT_PASS',err,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','preflight','train']);p.add_argument('--stop',type=int,default=3000);p.add_argument('--resume',type=Path);a=p.parse_args();out=ROOT/'runs'/RUN/'native-mid-fp32'
    if a.action=='prepare':prepare(out);return
    configure_workspace();contract_guard();set_reproducible(2026100438)
    if a.action=='preflight':preflight(out);return
    assert json.loads((out/'preflight.json').read_text())['status']=='PASS';tr=Training(out)
    if a.resume:tr.restore(torch.load(a.resume,map_location='cpu',weights_only=True))
    else:assert not (out/'fp32_metrics.jsonl').exists()
    with (out/'fp32_metrics.jsonl').open('a',encoding='utf-8',buffering=1) as f:
        while tr.step<min(a.stop,tr.cfg['fp32_steps']):
            row=tr.update();f.write(json.dumps(row)+'\n')
            if row['native']:gc.collect();torch.cuda.empty_cache()
            if tr.step%50==0:print('INTERMEDIATE_TRAIN',tr.step,round(row['loss'],4),round(row['flat']*128,3),round(row['phase']*128,3),flush=True)
            if tr.step%500==0 or tr.step==a.stop:
                torch.save(tr.state(),out/f'fp32_checkpoint_{tr.step:06d}.pt');tr.evaluate();gc.collect();torch.cuda.empty_cache()
            if (out/'stop_after_update').exists():torch.save(tr.state(),out/f'fp32_checkpoint_{tr.step:06d}.pt');return
    print('INTERMEDIATE_DONE',tr.step,flush=True)

if __name__=='__main__':main()
