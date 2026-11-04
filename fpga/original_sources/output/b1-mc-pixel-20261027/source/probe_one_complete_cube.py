"""Bounded positive capacity control: one 56px face, one 28px depth vector.

This coefficient construction does NOT assign cuboids to arbitrary real scenes.
It deliberately tests the corrected full-volume target before more training.
"""
from core import *
from train_pixel import checkpoint
from pixel_targets import panel
import r10_mass_cuboids as r10
from audit_complete_cube_target import complete_cube
from r15_structure_guard_probe import fixtures

@torch.no_grad()
def seed():
    # Seven existing low-resolution DW stages sweep ONE front mask together.
    # No object-dependent depth is inferred; this is the bounded28px control.
    n=r10.seed(dict(polarity='dark',morph=''))
    for j in range(5):
        for branch in range(3):
            row=branch*5+j
            n.bias['down2__pw'][row]=63.5+126*(127.5-127.5/2)
    # One front(55), one side(29), one top(97) on background235 for neutral input.
    n.weight['rgb_head'][:,:3,0,0]=torch.tensor([-180.,-26.,68.],device='cuda')/127
    n.bias['rgb_head'].copy_(torch.tensor([107.,-20.,361.],device='cuda'))
    for key,w in n.weight.items():
        need=(w.abs().flatten(1).amax(1)*131072/127).clamp_min(64)
        getattr(n,key+'_m').copy_((2**need.log2().ceil()).clamp(64,131071).double()[None,:,None,None])
    return n

@torch.inference_mode()
def main(destname='r26-complete-cube-capacity',front_size=56,depth=28):
    setup(271163);d=OUT/destname;d.mkdir(exist_ok=True);n=seed()
    checkpoint(n,d/'constructed.pt',0,trained=False,kind='one complete cube bounded capacity control, not a scene style')
    raw=n.export();(d/'candidate.bin').write_bytes(raw);eng=IntegerEngine(raw)
    yy,xx=np.mgrid[:256,:256];mask=(xx>=88)&(xx<88+front_size)&(yy>=80)&(yy<80+front_size);records=[]
    target=complete_cube(mask,[55]*3,depth,[235]*3)
    for name in ['uniform','front_lighting','front_flat_patch']:
        a=np.full((256,256,3),235,np.uint8);a[mask]=55
        if name=='front_lighting':
            v=np.clip(35+(xx-88)*1.2,0,255).astype(np.uint8);a[mask]=np.repeat(v[...,None],3,2)[mask]
        if name=='front_flat_patch':a[96:120,104:128]=110
        x=tensor(a)[None];nodes,_=eng(x*128,True);o=rgb(nodes['rgb_head']/128)
        assert torch.equal(n(x)*128,nodes['rgb_head'])
        truth=(target[...,0]<200);pred=o.mean(2)<180
        iou=float((truth&pred).sum()/max(1,(truth|pred).sum()))
        # Exclude a4px boundary band when measuring flat face interiors.
        face_metrics={}
        for label,colour in [('front',55),('top',97),('side',29)]:
            m=cv2.erode((target[...,0]==colour).astype(np.uint8),np.ones((9,9),np.uint8)).astype(bool)
            z=o[m].astype(float);face_metrics[label]=dict(rgb_mean=z.mean(0).tolist(),rgb_std=z.std(0).tolist())
        records.append(dict(name=name,mask_iou=iou,target_mae=float(np.abs(o.astype(float)-target).mean()),face_interiors=face_metrics))
        for key,z in [('source',a),('target',target),('student',o)]:Image.fromarray(z).save(d/f'{name}_{key}.png')
        panel([a,target,o],['输入 / 已知单方块正面',f'完整立方体目标 / 纵深{depth}像素','实际固定B1整数 / 构造系数，未训练'],d/f'{name}_comparison.png',width=384)
    negative=[]
    for name,a,positive in fixtures():
        if positive:continue
        nodes,_=eng(tensor(a)[None]*128,True)
        fields=list(getattr(n,'capacity_face_channels',(1,)))
        f=float((nodes['up2_low_pw'][:,fields]>1).any(1).float().mean());negative.append(dict(name=name,extrusion_fraction=f))
    assert min(r['mask_iou'] for r in records)>.95
    plane_pass=all(max(f['rgb_std'])<1 for r in records for f in r['face_interiors'].values())
    save_json(d/'report.json',dict(front_native_size=front_size,depth_native_offset=[depth,-depth],records=records,lighting_negatives=negative,
        trained=False,constructed=True,ordinary_fixed_B1=True,inference_extra_operators=False,
        silhouette_only_pass=True,face_plane_uniformity_pass=plane_pass,positive_control_pass=plane_pass,complete_real_scene_MC_pass=False,
        coefficient_sha256=sha(d/'candidate.bin'),reason='A single shared binary object front can make a full local cube; assigning that front and consistent scale to natural scenes remains unsolved. Unguarded lighting negatives still fail.'))
    print('ONE_COMPLETE_CUBE_CAPACITY',json.dumps(dict(records=records,negatives=negative)),flush=True)

if __name__=='__main__':main()
