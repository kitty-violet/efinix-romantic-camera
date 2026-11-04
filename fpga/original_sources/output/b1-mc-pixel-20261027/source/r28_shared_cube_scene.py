"""Correct shared top/side geometry in the existing R25 scene channels."""
from core import *
from r4_box_projector import BODY
from train_pixel import checkpoint
from pixel_targets import panel
from r15_structure_guard_probe import fixtures

@torch.no_grad()
def seed():
    n=load_model(OUT/'exports/mc-geometry-only-r25/selected.pt')
    # F rows0..4 remain shared with copies5..9(top),10..14(side).
    # One consensus before edge extraction; five later low stages sweep16px.
    for i,base in enumerate(BODY):
        w=n.weight[base+'__dw'];b=n.bias[base+'__dw'];pw=n.weight[base+'__pw'];pb=n.bias[base+'__pw']
        w[:15].zero_();b[:15].zero_();pw[:15].zero_();pb[:15].zero_()
        for ch in range(15):pw[ch,ch,0,0]=1
        if i==0:
            w[:15].fill_(1);b[:15]=-4*127
        else:
            w[:5,0,1,1]=1;w[5:15,0,1,1]=1
            if i==1:
                w[5:10,0,0,1]=-1;w[10:15,0,1,2]=-1
            else:w[5:15,0,2,0]=1
    w=n.weight['up1_low_dw'];b=n.bias['up1_low_dw'];w[:15].zero_();b[:15].zero_();w[:5,0,1,1]=1
    w[5:15,0,1,1]=1;w[5:15,0,2,0]=1
    w=n.weight['up1_low_pw'];b=n.bias['up1_low_pw'];w[:3].zero_();b[:3].zero_()
    w[0,:5,0,0]=.2
    w[1,5:10,0,0]=.2;w[1,:5,0,0]=-.2
    w[2,10:15,0,0]=.2;w[2,:10,0,0]=-.2
    # Existing hinges limit BOTH shared faces by the learned G field.
    w=n.weight['up2_low_pw'];b=n.bias['up2_low_pw'];w.zero_();b.zero_()
    for row,col in [(0,0),(1,1),(3,2),(5,4),(6,5)]:w[row,col,0,0]=1
    w[2,1,0,0]=1;w[2,3,0,0]=-1
    w[4,2,0,0]=1;w[4,3,0,0]=-1
    # F keeps the input's tone order. The top/side modify neighbour planes only
    # when G permits, with their shared geometric intersection already fixed.
    w=n.weight['rgb_head'];w.zero_()
    w[:,:5,0,0]=torch.tensor([-58.,-39.,39.,-67.,67.],device='cuda')[None]/32
    w[:,5:7,0,0]=torch.tensor([[2.,-2.],[0.,2.],[-2.,-2.]],device='cuda')
    n.bias['rgb_head'].copy_(torch.tensor([121.5,-5.5,375.5],device='cuda'))
    for key,w in n.weight.items():
        need=(w.abs().flatten(1).amax(1)*131072/127).clamp_min(64)
        # Never requantise learned support fields; preserve their integer rows.
        old=getattr(n,key+'_m').clone();new=(2**need.log2().ceil()).clamp(64,131071).double()[None,:,None,None]
        if key.startswith(('res1_','res2_','res3_')) or key in ['up1_low_dw']:
            old[:,:15]=new[:,:15]
        elif key=='up1_low_pw':old[:,:3]=new[:,:3]
        elif key in ['up2_low_pw','rgb_head']:old=new
        getattr(n,key+'_m').copy_(old)
    n.rgb_head_m.fill_(4096)
    return n

@torch.inference_mode()
def main():
    setup(271171);d=OUT/'r28-shared-cube-scene';d.mkdir(exist_ok=True);n=seed();raw=n.export();eng=IntegerEngine(raw)
    (d/'candidate.bin').write_bytes(raw);checkpoint(n,d/'initial.pt',0,trained=False,kind='corrected shared cube scene initializer')
    old=IntegerEngine((OUT/'exports/mc-geometry-only-r25/coefficients-padded.bin').read_bytes());rows=[]
    cases=[('corridor',read(OUT/'data-quality/VID_20260924_231158_source.png'),None),('outdoor',read(OUT/'targets/outdoor_source.png'),None),*fixtures()]
    for name,a,pos in cases:
        x=tensor(a)[None]*128;nodes,_=eng(x,True);o=rgb(nodes['rgb_head']/128);b=rgb(old(x)/128)
        assert torch.equal(n(x/128)*128,nodes['rgb_head'])
        f=nodes['up2_low_pw'];extra=((f[:,1]-f[:,2]).clamp_min(0)+(f[:,3]-f[:,4]).clamp_min(0))>1
        rows.append(dict(case=name,positive=pos,face_fraction=float(extra.float().mean())))
        Image.fromarray(o).save(d/f'{name}_student.png');panel([a,b,o],['原图','R25原三面','共享顶/侧交界 / B1构造候选'],d/f'{name}_comparison.jpg',width=480)
    save_json(d/'report.json',dict(records=rows,trained=False,inference_extra_operators=False,real_scene_MC_pass=False,changes='complete shared top/side intersection rather than D-H split'))
    print('SHARED_CUBE_SCENE',json.dumps(rows),flush=True)

if __name__=='__main__':main()
