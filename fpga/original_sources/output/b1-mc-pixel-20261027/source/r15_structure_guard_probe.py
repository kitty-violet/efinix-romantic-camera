"""Counterexample-driven structural guard prototype. Research only, not MC PASS.

Existing stem channels detect chromatic high-frequency boundaries. A spare
32-channel field spreads bounded structural support. Existing PW/ReLU hinges
limit extrusion and face fields by that support, including cancelling the
volume displacement where support is absent. No new graph operators.
"""
from core import *
from train_pixel import checkpoint
from pixel_targets import panel
from r4_box_projector import BODY

@torch.no_grad()
def seed():
    n=load_model(OUT/'r13-region-cuboids/dark-aligned-close/selected.pt')
    lap=torch.full((3,3),-1/8,device='cuda');lap[1,1]=1
    vectors=[[.5,0,-.5],[-.25,.5,-.25]]
    for j,vector in enumerate(vectors):
        for sign in [1,-1]:
            row=3+2*j+int(sign<0);n.weight['stem'][row].zero_()
            for ch,v in enumerate(vector):n.weight['stem'][row,ch].copy_(lap*v*sign)
            n.bias['stem'][row]=-3
            n.weight['down1__dw'][row].fill_(1);n.bias['down1__dw'][row]=0
    n.weight['down1__pw'][3].zero_();n.weight['down1__pw'][3,3:7,0,0]=1;n.bias['down1__pw'][3]=0
    n.weight['down2__dw'][3].fill_(1);n.bias['down2__dw'][3]=0
    n.weight['down2__pw'][31].zero_();n.weight['down2__pw'][31,3,0,0]=1;n.bias['down2__pw'][31]=-8
    for base in BODY:
        n.weight[base+'__dw'][31].fill_(1);n.bias[base+'__dw'][31]=0
        n.weight[base+'__pw'][31].zero_();n.weight[base+'__pw'][31,31,0,0]=1;n.bias[base+'__pw'][31]=0
    n.weight['up1_low_dw'][31].fill_(1);n.bias['up1_low_dw'][31]=0
    w=n.weight['up1_low_pw'];b=n.bias['up1_low_pw'];w.zero_();b.zero_()
    w[0,10:15,0,0]=.2 # volume D
    w[1,10:15,0,0]=.2;w[1,:5,0,0]=-.2 # E=D-F
    w[2,10:15,0,0]=.2;w[2,5:10,0,0]=-.2 # T=D-H
    w[3,31,0,0]=1 # structural support G
    from r3_palette_blocks import CHROMA_VALUES
    for group in range(2):
        w[4+group,15+group*8:23+group*8,0,0]=torch.tensor(np.diff(CHROMA_VALUES)/(2*127),device='cuda')
        b[4+group]=float((CHROMA_VALUES[0]+127)/2)
    w=n.weight['up2_low_pw'];b=n.bias['up2_low_pw'];w.zero_();b.zero_()
    for row,col in [(0,0),(1,1),(3,2),(5,4),(6,5)]:w[row,col,0,0]=1
    w[2,1,0,0]=1;w[2,3,0,0]=-1 # rejected extrusion max(E-G,0)
    w[4,2,0,0]=1;w[4,3,0,0]=-1 # rejected top max(T-G,0)
    w=n.weight['rgb_head'];b=n.bias['rgb_head'];w.zero_()
    # Exact integer identities: WD+WE+WP=0 and WT+WQ=0. When G=0,
    # all extrusion and invented face terms cancel, leaving the front F.
    q=torch.tensor([-58,-26,84,49,-49],device='cuda',dtype=torch.float32)
    w[:,:5,0,0]=q[None]/32;w[:,5:7,0,0]=torch.tensor([[2,-2],[0,2],[-2,-2]],device='cuda')
    b.copy_(torch.tensor([121.5,-5.5,375.5],device='cuda'))
    for key,weights in n.weight.items():
        need=(weights.abs().flatten(1).amax(1)*131072/127).clamp_min(64)
        getattr(n,key+'_m').copy_((2**need.log2().ceil()).clamp(64,131071).double()[None,:,None,None])
    n.rgb_head_m.fill_(4096)
    return n

def fixtures():
    yy,xx=np.mgrid[:256,:256];r2=(xx-127.5)**2+(yy-127.5)**2;spot=np.exp(-r2/(2*22**2))
    grey=np.clip(45+175*spot,0,255).astype(np.uint8);yield 'flat_neutral_spot',np.repeat(grey[...,None],3,2),False
    shadow=np.where(xx+yy<260,65,170).astype(np.uint8);yield 'flat_neutral_shadow',np.repeat(shadow[...,None],3,2),False
    coloured=np.clip(np.array([70,110,145])[None,None]+135*spot[...,None],0,255).astype(np.uint8);yield 'flat_colour_wall_spot',coloured,False
    mask=(xx>=88)&(xx<152)&(yy>=80)&(yy<144)
    greybox=np.where(mask[...,None],55,235).astype(np.uint8).repeat(3,2);yield 'grey_box_ambiguous',greybox,True
    colourbox=np.where(mask[...,None],np.array([45,85,165]),np.array([230,230,230])).astype(np.uint8);yield 'colour_box_positive',colourbox,True

@torch.inference_mode()
def main():
    setup(271091);d=OUT/'mc-structure-illumination-audit';d.mkdir(exist_ok=True);disk_guard();n=seed();raw=n.export()
    (d/'guard-prototype.bin').write_bytes(raw);checkpoint(n,d/'guard-prototype.pt',0,trained=False,kind='chromatic structural support counterexample prototype')
    old=IntegerEngine((OUT/'r13-region-cuboids/dark-aligned-close/candidate.bin').read_bytes());new=IntegerEngine(raw);records=[]
    for name,a,positive in fixtures():
        p,ps=old(tensor(a)[None]*128,True);q,qs=new(tensor(a)[None]*128,True)
        baseline=rgb(p['rgb_head']/128);guard=rgb(q['rgb_head']/128)
        assert torch.equal(n(tensor(a)[None])*128,q['rgb_head'])
        old_fraction=float((p['up2_low_pw'][:,1]>1).float().mean())
        allowed=(q['up2_low_pw'][:,1]-q['up2_low_pw'][:,2]).clamp_min(0)
        new_fraction=float((allowed>1).float().mean());support=float((q['up1_low_pw'][:,3]>1).float().mean())
        record=dict(id=name,positive_geometry_control=positive,old_extrusion_fraction=old_fraction,
            guarded_extrusion_fraction=new_fraction,structural_support_fraction=support,
            false_geometry_should_be_zero=not positive,neutral_geometry_false_negative_expected=name=='grey_box_ambiguous')
        if not positive:assert new_fraction==0,(name,new_fraction)
        if name=='colour_box_positive':assert new_fraction>.005
        panel([a,baseline,guard],['输入 / 已知平面或正例','旧：亮度挤出','结构支持原型 / 非最终MC'],d/f'{name}.png',width=400)
        records.append(record);print('ILLUMINATION_COUNTEREXAMPLE',json.dumps(record),flush=True)
    # Same pixels cannot distinguish a painted patch from a head-on box.
    # Preserve this known ambiguity instead of hiding the missing geometry.
    save_json(d/'counterexamples.json',dict(records=records,coefficient_sha256=sha(d/'guard-prototype.bin'),
        result='reject unguided luminance extrusion; guard passes selected lighting negatives and coloured positive, but misses neutral geometry',
        complete_geometry_recovery=False,visual_MC_accepted=False,inference_extra_operators=False,
        interpretation='necessary conservative gate only; chromatic boundaries can be painted material, not physical geometry'))
    print('STRUCTURE_GUARD_COUNTEREXAMPLES_COMPLETE',flush=True)

if __name__=='__main__':main()
