"""Train-only texture discrimination for the unchanged compact RGB graph."""
import copy
import json
from pathlib import Path
import shutil
import sys
import torch
from . import train_stroke_color as base
from .train_stroke_color_input import Training as ColorTraining
from .stroke_color_input_net import StrokeColorInputNet
from .train_stroke_color_stable import reference_warp
from .train_clean_capacity import TextureCritic
from .audit_capacity_texture import edge_map
from .runtime import file_hash,write_json,configure_workspace,contract_guard
from .engine import set_reproducible

ORIGINAL_SPATIAL=base.spatial


class Training(ColorTraining):
    def __init__(self,out):
        super().__init__(out)
        self.critic=TextureCritic().cuda().train();self.dopt=torch.optim.Adam(self.critic.parameters(),lr=.00015,betas=(.5,.99))
        self.in_update=False;self.critic_stats={};base.spatial=self.spatial_with_critic

    def spatial_with_critic(self,y,target,margin):
        losses=ORIGINAL_SPATIAL(y,target,margin)
        if not self.in_update:return losses
        side=128
        positions=[(self.rng.randrange(margin,y.shape[-2]-margin-side+1),self.rng.randrange(margin,y.shape[-1]-margin-side+1)) for _ in range(6)]
        with torch.no_grad():
            error=(edge_map(y)-edge_map(target)).abs()
            scores=[float(error[...,yy:yy+side,xx:xx+side].mean()) for yy,xx in positions]
        yy,xx=positions[max(range(len(scores)),key=scores.__getitem__)]
        weights=y.new_tensor([self.rng.uniform(.2,.4),self.rng.uniform(.45,.65),self.rng.uniform(.05,.2)]);weights/=weights.sum()
        gray=lambda z:(z[...,yy:yy+side,xx:xx+side]*weights[None,:,None,None]).sum(1,keepdim=True)
        real,fake=gray(target),gray(y)
        self.critic.requires_grad_(True);self.critic.train();self.dopt.zero_grad(set_to_none=True)
        dr,df=self.critic(real),self.critic(fake.detach());d=.5*((dr-1).square().mean()+df.square().mean())
        assert torch.isfinite(d);d.backward();self.dopt.step()
        self.critic.requires_grad_(False);self.critic.eval()
        adv=(self.critic(fake)-1).square().mean()
        losses['adversarial']=min(1,(self.step+1)/300)*adv
        self.critic_stats=dict(critic_loss=float(d.detach()),raw_adversarial=float(adv.detach()),critic_xy=[xx,yy])
        return losses

    def update(self,native=None):
        self.in_update=True
        try:row=super().update(native)
        finally:self.in_update=False
        row.update(self.critic_stats);return row

    def state(self):
        s=super().state();s.update(critic=self.critic.state_dict(),critic_optimizer=self.dopt.state_dict());return s

    def restore(self,s):
        super().restore(s);self.critic.load_state_dict(s['critic']);self.dopt.load_state_dict(s['critic_optimizer'])


def prepare(out,steps):
    source=base.ROOT/'native-stroke-color-stable-v3';out.mkdir(exist_ok=False)
    shutil.copy2(source/'data.json',out/'data.json');shutil.copytree(source/'source',out/'source')
    cfg=json.loads((source/'config.json').read_text(encoding='utf-8'));initial=source/'checkpoint_000800.pt'
    cfg.update(initial=str(initial),initial_sha256=file_hash(initial),steps=steps,
        critic='train-only gray 128x128 LSGAN, highest edge-error patch among6 positions, actual teacher targets',critic_deployed=False)
    cfg['weights'].update(adversarial=1.2,clean=12)
    write_json(out/'config.json',cfg);b=json.loads((source/'binding.json').read_text())
    assert all(file_hash(p)==h for p,h in b['code'].items())
    for p in [Path(__file__),Path(__file__).with_name('train_clean_capacity.py')]:
        b['code'][str(p)]=file_hash(p);shutil.copy2(p,out/'source'/p.name)
    b['config']=file_hash(out/'config.json');b['data']=file_hash(out/'data.json');write_json(out/'binding.json',b)
    print('TEXTURE_COLOR_PREPARED',flush=True)


def active_check(out):
    configure_workspace();contract_guard();set_reproducible(2026100487);torch.set_num_threads(4)
    t=Training(out);t.step=500;t.update(native=False);torch.save(t.state(),out/'active_temporal_preflight.pt')
    ref=t.update(native=False);expected=copy.deepcopy(t.net.state_dict());dc=copy.deepcopy(t.critic.state_dict())
    t.restore(torch.load(out/'active_temporal_preflight.pt',map_location='cpu',weights_only=True));replay=t.update(native=False)
    error=max(float((v-expected[k]).abs().max()) for k,v in t.net.state_dict().items())
    derror=max(float((v-dc[k]).abs().max()) for k,v in t.critic.state_dict().items())
    passed=error<=1e-7 and derror<=1e-7 and ref['sample']==replay['sample'] and ref['critic_xy']==replay['critic_xy']
    write_json(out/'active_temporal_resume.json',dict(status='PASS' if passed else 'FAIL',resume_max_error=error,critic_resume_max_error=derror,
        reference=ref,replay=replay,diagnostic_schedule_step=500,actual_formal_training_steps=0,binding=t.binding))
    assert passed
    print('TEXTURE_ACTIVE_RESTORE_PASS',error,derror,flush=True)


def main():
    base.StrokeColorNet=StrokeColorInputNet;base.Training=Training;base.warp=reference_warp;base.prepare=prepare
    if '--name' not in sys.argv:sys.argv.extend(['--name','native-stroke-color-texture-v4'])
    out=base.ROOT/sys.argv[sys.argv.index('--name')+1]
    if sys.argv[1]=='active-check':active_check(out);return
    if sys.argv[1]=='train':
        gate=json.loads((out/'active_temporal_resume.json').read_text());assert gate['status']=='PASS' and gate['binding']==json.loads((out/'binding.json').read_text())
    base.main()


if __name__=='__main__':main()
