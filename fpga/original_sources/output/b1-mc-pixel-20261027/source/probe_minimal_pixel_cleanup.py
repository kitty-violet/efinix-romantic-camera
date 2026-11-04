"""Inspect/clean chroma fringe masks only; retain A's exact Y geometry/readout."""
from core import *
from train_pixel import checkpoint
from pixel_targets import panel
from evaluate_r3_r4 import controls
from r13_region_cuboids import region_stats

@torch.no_grad()
def seed(strength):
    n=load_model(OUT/'exports/pixel-user-curve-A/selected.pt')
    if strength=='original':return n
    # Preserve all luminance masks, face fields, palette/readout and multipliers.
    # Centre-heavy binary voting can remove an isolated fringe cell, but cannot
    # grow whole dark/light regions as rejected R23 did.
    key='res2_conv1__dw';centre=7 if strength=='isolated-only' else 5
    n.weight[key][8:24].fill_(1);n.weight[key][8:24,0,1,1]=centre
    n.bias[key][8:24]=-((centre+8)//2)*127
    need=(n.weight[key].abs().flatten(1).amax(1)*131072/127).clamp_min(64)
    getattr(n,key+'_m').copy_((2**need.log2().ceil()).clamp(64,131071).double()[None,:,None,None])
    return n

@torch.inference_mode()
def main():
    setup(271170);root=OUT/'r28-pixel-minimal-cleanup';root.mkdir(exist_ok=True)
    old=IntegerEngine((OUT/'exports/pixel-user-curve-A/coefficients-padded.bin').read_bytes());reports=[];imgs={}
    cases={s:read(OUT/f'data-quality/{v}_source.png') for s,v in [('corridor','VID_20260924_231158'),('outdoor','VID_20260925_070032'),('indoor_training','VID_20260925_070939')]}
    for name in ['original','isolated-only','fringe-two-neighbours']:
        d=root/name;d.mkdir(exist_ok=True);n=seed(name);raw=n.export();(d/'candidate.bin').write_bytes(raw);checkpoint(n,d/'constructed.pt',0,trained=False,cleanup=name)
        eng=IntegerEngine(raw);rows=[];imgs[name]={}
        for scene,a in cases.items():
            x=tensor(a)[None]*128;z,_=eng(x,True);baseline,_=old(x,True)
            for node in ['down2.pw','res1_conv1.pw','res1_conv2.pw','res2_conv1.pw','res2_conv2.pw','res3_conv1.pw']:
                assert torch.equal(z[node][:,:8],baseline[node][:,:8]),(name,node)
            assert torch.equal(z['res3_conv2.pw'][:,[0,3,4]],baseline['res3_conv2.pw'][:,[0,3,4]])
            assert torch.equal(n(x/128)*128,z['rgb_head'])
            o=rgb(z['rgb_head']/128);b=rgb(baseline['rgb_head']/128);delta=np.abs(o.astype(float)-b)
            masks=baseline['down2.pw'][:,8:24];fraction=float(((masks>0)&(masks<127)).float().mean())
            rows.append(dict(scene=scene,changed_pixel_fraction=float((delta.max(2)>0).mean()),mean_rgb_change=float(delta.mean()),
                original_soft_chroma_gate_fraction=fraction,old_regions=region_stats(b),new_regions=region_stats(o)))
            Image.fromarray(o).save(d/f'{scene}_student.png');imgs[name][scene]=o
        gray=controls(dict(candidate=eng))['candidate']['gray_curve_mean'];ref=controls(dict(original=old))['original']['gray_curve_mean']
        assert np.array_equal(gray,ref)
        save_json(d/'report.json',dict(cleanup=name,records=rows,gray_curve_A_exact=True,Y_geometry_unchanged=True,saturation=1.3,trained=False,source_checkpoint=str(OUT/'exports/pixel-user-curve-A/selected.pt')))
        reports.append(dict(name=name,records=rows));print('MINIMAL_PIXEL_CLEANUP',name,json.dumps(rows),flush=True)
    for scene,a in cases.items():
        panel([a,*[imgs[k][scene] for k in imgs]],['原图','原A / 保留块面','只去孤立杂色','边缘杂色轻清理'],root/f'{scene}_comparison.jpg',width=480)
        h,w=a.shape[:2];box=(slice(72,232),slice(200,456)) if scene=='corridor' else (slice(100,260),slice(650,906))
        panel([imgs[k][scene][box] for k in imgs],['原A同一局部','只去孤立杂色','边缘杂色轻清理'],root/f'{scene}_crop.png',width=512)
    save_json(root/'summary.json',reports)

if __name__=='__main__':main()
