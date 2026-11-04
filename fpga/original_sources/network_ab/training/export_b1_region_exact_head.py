"""Use an exact dyadic Q17 scale for the known RGB colour conversion matrix."""
from pathlib import Path
import shutil
import torch
from . import train_b1_luma_regions as region
from . import train_b1_learned_colour as colour
from .runtime import configure_workspace,contract_guard,write_json,file_hash
from .engine import set_reproducible


def main():
    configure_workspace();set_reproducible(2026100525);contract_guard()
    source=region.OUT/'qat/checkpoint_003500.pt';out=region.OUT/'exact-head';out.mkdir(exist_ok=False)
    data,selected,cache=colour.load_data();tr=region.Trainer(cache,selected,3000)
    tr.net=region.integer_version(tr.net);tr.new_optimizer();tr.restore(source)
    w,b=tr.net.weight['rgb_head'],tr.net.bias['rgb_head']
    assert (w*2).abs().max()<=127 and torch.equal(w*2,(w*2).round()) and torch.equal(b*2,(b*2).round())
    rng=torch.Generator(device='cpu').manual_seed(123)
    features=torch.randint(0,2,(4096,8),generator=rng).cuda().float()
    features[:,6:]=torch.randint(0,65,(4096,2),generator=rng).cuda().float()
    ideal=(features@w[:,:,0,0].T+b).round().clamp(-128,127)
    def replay():
        m=tr.net.rgb_head_m.flatten();s=(m/131072).float()
        wi=tr.net.round(w[:,:,0,0]/s[:,None]).clamp(-127,127)
        bi=tr.net.round(b/s)
        return tr.net.round((features@wi.T+bi).double()*m/131072).clamp(-128,127).float()
    before=replay();old_m=tr.net.rgb_head_m.flatten().tolist()
    with torch.no_grad():tr.net.rgb_head_m.fill_(65536)
    after=replay();error=float((after-ideal).abs().max());assert error==0
    tr.save(out/'checkpoint_003500_exact_head.pt')
    region.review(tr,data,selected,out/'package')
    write_json(out/'proof.json',dict(source=str(source),source_sha256=file_hash(source),old_multiplier=old_m,new_multiplier=65536,
        random_colour_cases=4096,before_max_error=float((before-ideal).abs().max()),before_rgb_bias=(before-ideal).mean(0).tolist(),
        after_max_error=error,reason='RGB matrix is multiples of 4, use exact 1/2 weight scale instead of max/112',
        learned_encoder_unchanged=True,postfilter=False,actual_board=False))
    shutil.copy2(__file__,out/Path(__file__).name);print('EXACT_HEAD_COMPLETE',float((before-ideal).abs().max()),error,flush=True)


if __name__=='__main__':main()
