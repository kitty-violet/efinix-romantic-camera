"""Requested 120% luma contrast and 130% chroma, encoded in B1 readout."""
from core import *
from train_pixel import checkpoint
from pixel_targets import panel

@torch.no_grad()
def main():
    setup();d=OUT/'pixel-grade-c120-s130';d.mkdir(exist_ok=True);disk_guard()
    src=OUT/'r3-shared-palette/flat-large/initial.pt';n=load_model(src);contrast=1.2;saturation=1.3
    original=n.export();w=n.weight['rgb_head'];b=n.bias['rgb_head'];centre=w[:,1:3,0,0].sum(1)*63.5
    for col in [0,3,4]:w[:,col].mul_(contrast)
    w[:,1:3].mul_(saturation);b.sub_(128*(contrast-1)+(saturation-1)*centre)
    for key,weights in n.weight.items():
        need=(weights.abs().flatten(1).amax(1)*131072/127).clamp_min(64)
        getattr(n,key+'_m').copy_((2**need.log2().ceil()).clamp(64,131071).double()[None,:,None,None])
    raw=n.export();(d/'candidate.bin').write_bytes(raw);checkpoint(n,d/'selected.pt',0,trained=False,
        kind='user requested grading inside ordinary B1',contrast=contrast,saturation=saturation,
        source_checkpoint=str(src),separate_tone_curve_edit=False)
    engine=IntegerEngine(raw);old=IntegerEngine(original)
    for scene in ['outdoor','indoor']:
        a=read(OUT/f'targets/{scene}_source.png');o=rgb(engine(tensor(a)[None]*128)/128);p=rgb(old(tensor(a)[None]*128)/128)
        assert torch.equal(n(tensor(a)[None])*128,engine(tensor(a)[None]*128))
        Image.fromarray(o).save(d/f'{scene}_student.png')
        panel([a,p,o],['原图','认可原版 C100/S100','实际B1 C120/S130 / 未另改亮度曲线'],d/f'{scene}_comparison.jpg')
    from evaluate_r3_r4 import controls
    save_json(d/'grading.json',dict(contrast=contrast,saturation=saturation,controls=controls(dict(original=old,graded=engine)),
        source_sha256=hashlib.sha256(original).hexdigest(),coefficient_sha256=sha(d/'candidate.bin'),
        changed_fields='rgb_head luma/face contrast around128 and centred chroma gain; Q17 container recalculated',
        separate_tone_curve_edit=False,step_structure_preserved=True,inference_extra_operators=False,visual_accepted=False))
    print('REQUESTED_PIXEL_GRADE_CREATED',sha(d/'candidate.bin'),flush=True)

if __name__=='__main__':main()
