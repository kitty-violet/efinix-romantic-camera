"""Restore the actual accepted teacher as target; mine excess-texture patches.

Same clean-skip student and 20-KiB packing. No replacement region algorithm
provides training RGB targets here. Structure/texture constraints remain
train-only, with source-bounded temporal loss as before.
"""
import argparse,json,gc
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from . import train_clean_skip as runner
from . import train_clean_capacity as method
from .clean_skip_capacity_net import plan,budget
from .train_component_centers import spatial as previous_spatial,target_components
from .audit_capacity_texture import edge_map,texture_scores
from .component_region_net import decode
from .train_b1_distillation import read_image,as_tensor,to_rgb
from .train_luma_regions import ROOT,RUN
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible


def balanced_spatial(p,t,margin):
    values=previous_spatial(p,t,margin);c=target_components(t);terms=[];cut=lambda x:x[...,margin:-margin,margin:-margin]
    for d in (1,3,7):
        for axis in (-1,-2):
            a=[slice(None)]*4;b=a.copy();a[axis]=slice(None,-d);b[axis]=slice(d,None);a=tuple(a);b=tuple(b)
            dp,dt=p[a]-p[b],c[a]-c[b];edge=cut(dt.abs().mean(1,keepdim=True)>4/128)
            terms.append((cut((dp-dt).abs().mean(1,keepdim=True))*edge).sum()/edge.sum().clamp_min(1))
    values['boundary']=sum(terms)/len(terms);return values


class Training(runner.Training):
    def critic_images(self,a,b,m):
        w=a.new_tensor([self.rng.uniform(.2,.4),self.rng.uniform(.45,.65),self.rng.uniform(.05,.2)]);w/=w.sum();size=self.cfg['critic_patch']
        positions=[(self.rng.randrange(m,a.shape[2]-m-size+1),self.rng.randrange(m,a.shape[3]-m-size+1)) for _ in range(8)]
        with torch.no_grad():
            excess=F.relu(edge_map(a)-edge_map(b));scores=[float(excess[...,y:y+size,x:x+size].mean()) for y,x in positions]
        yy,xx=positions[max(range(len(scores)),key=scores.__getitem__)];self.critic_xy=[xx,yy]
        gray=lambda x:(x[:,:,yy:yy+size,xx:xx+size]*w[None,:,None,None]).sum(1,keepdim=True)
        return gray(a),gray(b)

    def update(self,native=None):
        previous=method.spatial;method.spatial=balanced_spatial
        try:r=super().update(native)
        finally:method.spatial=previous
        r['critic_xy']=self.critic_xy;return r

    @torch.no_grad()
    def evaluate(self):
        folder=self.out/f'fp32_{self.step:06d}';folder.mkdir(exist_ok=False);rows=[]
        for r in self.data['validation']:
            x=as_tensor(read_image(r['source']))[None].cuda();p=decode(self.net(x));teacher=as_tensor(read_image(r['target']))[None].cuda();t=decode(target_components(teacher))
            Image.fromarray(to_rgb(p[0])).save(folder/(r['id']+'.png'));rows.append(dict(id=r['id'],teacher_rgb_mae=float((p-teacher).abs().mean()*128),**texture_scores(p,t)))
        write_json(folder/'scores.json',dict(records=rows,not_blind=True,actual_teacher=True,style_accepted=False))
        board=Image.new('RGB',(1280,1192),'#18202a');d=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
        for j,(key,(x,y)) in enumerate([('val_stairs_090',(150,430)),('val_stairs_090',(424,320)),('val_plaza_090',(80,400)),('val_new_hallway_120',(700,280))]):
            r=next(r for r in self.data['validation'] if r['id']==key);base=ROOT/'runs'/RUN
            for i,(path,title) in enumerate([(r['source'],'原片'),(r['target'],'原教师目标'),(base/'b1-capacity-control-v2/capacity/fp32_002000'/(key+'.png'),'修改前'),(folder/(key+'.png'),f'教师＋清洁约束 {self.step}步')]):
                d.text((i*320+5,j*298+5),title,font=font,fill='white');board.paste(Image.open(path).crop((x,y,x+320,y+264)),(i*320,j*298+30))
        board.save(folder/'comparison.png')


def prepare(out,steps):
    runner.prepare(out,steps);old=json.loads((out/'data.json').read_text(encoding='utf-8'));manifest=json.loads((ROOT/'runs/b1-cartoon-budget-20261004/data_manifest.json').read_text(encoding='utf-8'));lookup={(r['source'],r['current_index']):r for r in manifest['pairs']}
    for r in old['pairs']:
        original=lookup[(r['source'],r['current_index'])];assert original['next_index']==r['next_index'] and original['current_sha256']==r['current_sha256'] and original['split']=='train'
        for k in ['current_target','next_target']:
            r[k]=original[k];r[k+'_sha256']=original[k+'_sha256'];assert file_hash(r[k])==r[k+'_sha256']
    old['target_kind']='actual_original_teacher_not_L0_proxy';write_json(out/'data.json',old)
    cfg=json.loads((out/'config.json').read_text());cfg.update(method='Actual original teacher + balanced gradient edges + clean skip features + mined texture LSGAN',actual_teacher=True,critic_patch_mining_candidates=8)
    cfg['objective_weights'].update(boundary=2,adversarial=.2)
    write_json(out/'config.json',cfg);binding=json.loads((out/'binding.json').read_text());binding['code'][str(Path(__file__))]=file_hash(__file__);binding.update(config_sha256=file_hash(out/'config.json'),data_sha256=file_hash(out/'data.json'));write_json(out/'binding.json',binding)
    t=torch.zeros(1,3,96,96);t[:,:,:,48:]=.5;c=target_components(t);blur=F.avg_pool2d(F.pad(c,(4,)*4,mode='replicate'),9,1)
    assert float(balanced_spatial(c,t,8)['boundary'])==0 and float(balanced_spatial(blur,t,8)['boundary'])>0
    write_json(out/'teacher_target_audit.json',dict(pairs=len(old['pairs']),targets_hashed=2*len(old['pairs']),training_sources=len({r['source'] for r in old['pairs']}),inputs_unchanged=True,validation_not_used_for_training=True,perfect_edge_loss=0,blur_edge_loss=float(balanced_spatial(blur,t,8)['boundary'])));print('ACTUAL_TEACHER_TARGETS_CHECKED',len(old['pairs'])*2,flush=True)


def main():
    import sys
    # Reuse the tested runner while explicitly binding this implementation in
    # every checkpoint. The runner's global factory is changed only here.
    runner.Training=Training;old_prepare=runner.prepare
    # prepare() above must call the ORIGINAL rather than recursively itself.
    def routed_prepare(out,steps):
        runner.prepare=old_prepare
        try:prepare(out,steps)
        finally:runner.prepare=routed_prepare
    runner.prepare=routed_prepare
    if '--name' not in sys.argv:sys.argv.extend(['--name','teacher-clean-skip-v1'])
    runner.main()


if __name__=='__main__':main()
