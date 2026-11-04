"""Learn from a nonlinear rank seed; every inference operation remains B1."""
import argparse
import json
import shutil
from pathlib import Path

import torch

from . import train_b1_learned_colour as colour
from . import train_b1_colour_texture as texture
from .seed_b1_rank_blocks import OUT as SEEDS
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .engine import set_reproducible

OUT=SEEDS/'learned'


def make(cache,selected,steps):
    tr=texture.Trainer(cache,selected,steps)
    tr.net=colour.ColourB1(revive=False,source=SEEDS/'blocks/coefficients-padded.bin').cuda().eval()
    tr.contour_weight=0.  # channel24 now carries a rank colour, not a contour.
    groups=[]
    for kind,parameters in [('w',tr.net.weight),('b',tr.net.bias)]:
        for key,p in parameters.items():
            rate=.00025 if kind=='w' else .01
            groups.append(dict(params=[p],lr=rate,initial_lr=rate))
    tr.opt=torch.optim.Adam(groups)
    return tr


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=2000);a=ap.parse_args()
    configure_workspace();set_reproducible(2026100519);contract_guard();OUT.mkdir(exist_ok=False)
    data,selected,cache=colour.load_data();tr=make(cache,selected,a.steps)
    write_json(OUT/'binding.json',dict(seed=str(SEEDS/'blocks/coefficients-padded.bin'),
        seed_sha256=file_hash(SEEDS/'blocks/coefficients-padded.bin'),steps=a.steps,selected=selected,
        code={str(p):file_hash(p) for p in [Path(__file__),Path(colour.__file__),Path(texture.__file__)]},
        actual_B1=True,oracle_contour_injection=False,all_colour_weights_trainable=True,
        auxiliary_contour_loss=False,critic_deployed=False))
    for src in (__file__,colour.__file__,texture.__file__):shutil.copy2(src,OUT/Path(src).name)
    native=tr.update(True);crop=tr.update();tr.save(OUT/'resume_probe.pt');ref=tr.update()
    expected={k:v.detach().clone() for k,v in tr.net.state_dict().items()}
    expected_d={k:v.detach().clone() for k,v in tr.critic.state_dict().items()}
    fresh=make(cache,selected,a.steps);fresh.restore(OUT/'resume_probe.pt');got=fresh.update()
    error=max(float((v-expected[k]).abs().max()) for k,v in fresh.net.state_dict().items())
    derror=max(float((v-expected_d[k]).abs().max()) for k,v in fresh.critic.state_dict().items())
    assert error<1e-5 and derror<1e-5 and ref['crop']==got['crop']
    write_json(OUT/'preflight.json',dict(status='PASS',native=native,crop=crop,resume_error=error,critic_resume_error=derror))
    print('RANK_LEARNING_PREFLIGHT_PASS',error,derror,flush=True)
    del tr,fresh;torch.cuda.empty_cache();torch.manual_seed(2026100519);torch.cuda.manual_seed_all(2026100519)
    tr=make(cache,selected,a.steps)
    colour.review(tr,data,selected,OUT/'initial',oracle_comparison=False)
    with (OUT/'metrics.jsonl').open('x',encoding='utf-8',buffering=1) as f:
        while tr.step<a.steps:
            row=tr.update();f.write(json.dumps(row)+'\n')
            if tr.step%100==0:print('RANK_LEARN',tr.step,round(row['loss'],4),round(row['discriminator'],4),flush=True)
            if tr.step%500==0 or tr.step==a.steps:
                tr.save(OUT/f'checkpoint_{tr.step:06d}.pt');colour.review(tr,data,selected,OUT/f'step_{tr.step:06d}',oracle_comparison=False)
    print('RANK_LEARNING_COMPLETE',flush=True)


if __name__=='__main__':main()
