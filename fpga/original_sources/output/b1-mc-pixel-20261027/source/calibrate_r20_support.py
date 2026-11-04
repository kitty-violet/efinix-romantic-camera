"""Calibrate the existing support readout; separately audit unseen fixtures."""
from core import *
from train_pixel import checkpoint
from pixel_targets import panel
from r18_learn_structure_support import samples
from r15_structure_guard_probe import fixtures
import argparse

@torch.inference_mode()
def predictions(n,arrays):
    x=torch.stack([tensor(a,'cpu') for a in arrays]).cuda();pred=[]
    for v in x.split(8):pred.append(n(v,True)['up1_low_pw'][:,3].cpu().numpy())
    return np.concatenate(pred)

def stats(pred,target,cut):
    p=pred[:,10:-10,10:-10]>cut;t=target[:,10:-10,10:-10]>.5
    return dict(true_positive_rate=float(p[t].mean()),false_positive_rate=float(p[~t].mean()))

def main(rootname='r20-multifeature-structure',test_seed=271115,min_hit=.6,prefer_precision=False,checkpoint_name='selected.pt'):
    setup(271123);root=OUT/rootname;d=root/'calibrated';d.mkdir(exist_ok=True);disk_guard()
    source_checkpoint=root/checkpoint_name;n=load_model(source_checkpoint);va,vy,vk=samples(96,271112);p=predictions(n,va);rows=[]
    for cut in range(128):
        s=stats(p,vy,cut);rows.append(dict(threshold=cut,**s))
    eligible=[r for r in rows if r['true_positive_rate']>=min_hit and r['false_positive_rate']<=.2]
    best=(min(eligible,key=lambda r:r['false_positive_rate']) if prefer_precision else max(eligible,key=lambda r:r['true_positive_rate'])) if eligible else min(rows,key=lambda r:3*(1-r['true_positive_rate'])+2*r['false_positive_rate'])
    cut=best['threshold'];gain=8
    with torch.no_grad():
        n.weight['up1_low_pw'][3].mul_(gain);n.bias['up1_low_pw'][3].sub_(cut).mul_(gain)
        need=max(64,float(n.weight['up1_low_pw'][3].abs().max())*131072/127)
        n.up1_low_pw_m[:,3]=min(131071,2**np.ceil(np.log2(need)))
    source_step=int(torch.load(source_checkpoint,map_location='cpu',weights_only=True)['step'])
    checkpoint(n,d/'selected.pt',source_step,trained=True,kind='MC structural support calibrated in existing PW coefficients',support_threshold=cut,support_gain=gain,source_checkpoint=str(source_checkpoint))
    raw=n.export();(d/'candidate.bin').write_bytes(raw);eng=IntegerEngine(raw)
    vp=predictions(n,va);validation=stats(vp,vy,1)
    # Fresh samples: no thresholds, gradients or checkpoint selection use these.
    ta,ty,tk=samples(192,test_seed);tp=predictions(n,ta);test=stats(tp,ty,1)
    cases=[];old=IntegerEngine((OUT/'r13-region-cuboids/dark-aligned-close/candidate.bin').read_bytes())
    for name,a,positive in fixtures():
        nodes,_=eng(tensor(a)[None]*128,True);o=rgb(nodes['rgb_head']/128);base=rgb(old(tensor(a)[None]*128)/128)
        assert torch.equal(n(tensor(a)[None])*128,nodes['rgb_head'])
        allowed=(nodes['up2_low_pw'][:,1]-nodes['up2_low_pw'][:,2]).clamp_min(0)
        frac=float((allowed>1).float().mean());s=float((nodes['up1_low_pw'][:,3]>1).float().mean())
        panel([a,base,o],['平面光照/真实形状对照','旧亮度挤出','学习支持+校准 / 实际B1'],d/f'{name}.png',width=400)
        cases.append(dict(fixture=name,positive=positive,extrusion_fraction=frac,support_fraction=s))
    for scene in ['outdoor','indoor']:
        a=read(OUT/f'targets/{scene}_source.png');nodes,_=eng(tensor(a)[None]*128,True);o=rgb(nodes['rgb_head']/128)
        Image.fromarray(o).save(d/f'{scene}_student.png')
        panel([a,o],['原图','实际B1 / 结构支持校准候选'],d/f'{scene}_comparison.jpg')
    negatives_pass=all(r['extrusion_fraction']==0 for r in cases if not r['positive'])
    positives_pass=all(r['extrusion_fraction']>.005 for r in cases if r['positive'])
    save_json(d/'calibration.json',dict(calibration_source_seed=271112,unseen_test_seed=test_seed,calibration_samples=96,test_samples=192,
        selected=best,validation_after_exact_quantization=validation,unseen_test=test,readout_gain=gain,counterexamples=cases,
        calibration_min_hit=min_hit,prefer_precision=prefer_precision,source_checkpoint=str(source_checkpoint),source_step=source_step,
        selected_lighting_negatives_pass=negatives_pass,positive_geometry_controls_pass=positives_pass,
        support_classification_pass=test['true_positive_rate']>=.6 and test['false_positive_rate']<=.25,
        complete_MC_accepted=False,inference_extra_operators=False,coefficient_sha256=sha(d/'candidate.bin'),all_calibration_thresholds=rows))
    print('MC_SUPPORT_CALIBRATED',json.dumps(dict(selected=best,validation=validation,unseen_test=test,counterexamples=cases)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='r20-multifeature-structure');p.add_argument('--test-seed',type=int,default=271115)
    p.add_argument('--min-hit',type=float,default=.6);p.add_argument('--prefer-precision',action='store_true');p.add_argument('--checkpoint',default='selected.pt')
    args=p.parse_args();main(args.root,args.test_seed,args.min_hit,args.prefer_precision,args.checkpoint)
