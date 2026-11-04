"""Bounded luminance sharpening entirely encoded in existing B1 channels."""
from core import *
from evaluate_r3_r4 import controls,content
from probe_pixel_colour_sharp import colour_stats

D=OUT/'pixel-r39-bounded-sharp';BASE=OUT/'exports/pixel-A-colour-sharp-r37/selected.pt'

@torch.no_grad()
def make(cap):
    n=load_model(BASE);amount=.375
    for row in [3,4]:
        n.weight['res3_conv2__pw'][row].copy_(n.weight['res3_conv2__pw'][0])
        n.bias['res3_conv2__pw'][row].copy_(n.bias['res3_conv2__pw'][0])
        n.res3_conv2__pw_m[:,row].copy_(n.res3_conv2__pw_m[:,0])
    lap=torch.zeros((3,3),device='cuda');lap[1,1]=4*amount
    for y,x in [(0,1),(1,0),(1,2),(2,1)]:lap[y,x]=-amount
    for row,sign in [(3,1),(4,-1)]:
        n.weight['up1_low_dw'][row,0]=sign*lap;n.bias['up1_low_dw'][row]=0;n.up1_low_dw_m[:,row]=2048
    # ReLU(p)-ReLU(p-cap) gives a capped positive detail response.
    for row,source in [(3,3),(4,4),(5,3),(6,4)]:
        n.weight['up1_low_pw'][row].zero_();n.weight['up1_low_pw'][row,source,0,0]=1
        n.bias['up1_low_pw'][row]=(-cap if row>=5 else 0);n.up1_low_pw_m[:,row]=2048
    for key in ['up1_high_dw','up2_low_dw']:
        for row in range(3,7):
            n.weight[key][row].zero_();n.weight[key][row,0,1,1]=1
            n.bias[key][row]=0;getattr(n,key+'_m')[:,row]=2048
    n.weight['up2_low_pw'][0].zero_();n.bias['up2_low_pw'][0]=0;n.up2_low_pw_m[:,0]=2048
    for col,value in [(0,1),(3,1),(4,-1),(5,-1),(6,1)]:n.weight['up2_low_pw'][0,col,0,0]=value
    return n

@torch.inference_mode()
def main():
    setup();D.mkdir(exist_ok=True);disk_guard();shutil.copy2(__file__,D/'source_used.py')
    mild=IntegerEngine((OUT/'exports/pixel-A-colour-sharp-r37/coefficients-padded.bin').read_bytes())
    prior=IntegerEngine((OUT/'exports/pixel-A-crisp-r36/coefficients-padded.bin').read_bytes())
    gray=controls({'mild':mild})['mild']['gray_curve_mean'];reports=[];images={}
    cases={'corridor':OUT/'data-quality/VID_20260924_231158_source.png','outdoor':OUT/'targets/outdoor_source.png',
           'indoor_training':OUT/'data-quality/VID_20260925_070939_source.png'}
    for label,cap in [('medium',3),('strong',6)]:
        dd=D/label;dd.mkdir(exist_ok=True);n=make(cap);raw=n.export();e=IntegerEngine(raw)
        (dd/'candidate.bin').write_bytes(raw)
        torch.save(dict(model=n.state_dict(),residual_encoding=copy.deepcopy(n.package['residual_encoding']),
            qat=True,step=0,trained=False,detail_cap=cap,laplacian_amount=.375,source=str(BASE)),dd/'constructed.pt')
        ctrl=controls({'e':e})['e'];assert ctrl['gray_sweep_max_channel_difference']==0 and np.array_equal(ctrl['gray_curve_mean'],gray)
        records=[];images[label]={}
        for scene,p in cases.items():
            a=read(p);x=tensor(a)[None]*128;nodes,_=e(x,True);ref,_=mild(x,True)
            assert torch.equal(nodes['res3_conv2.pw'][:,:3],ref['res3_conv2.pw'][:,:3])
            latent_error=float((nodes['up2_low_pw'][:,0]-ref['up2_low_pw'][:,0]).abs().max());assert latent_error<=cap
            assert torch.equal(n(x/128)*128,nodes['rgb_head'])
            z=rgb(nodes['rgb_head']/128);b=rgb(ref['rgb_head']/128);images[label][scene]=z
            rgb_error=float(np.abs(z.astype(float)-b).max());assert rgb_error<=np.ceil(cap*77/32)
            flat=z.reshape(180,4,320,4,3);assert np.all(flat.max((1,3))==flat.min((1,3)))
            Image.fromarray(z).save(dd/f'{scene}.png')
            edge=lambda v:float(np.abs(np.diff(v[::4,::4].astype(float).mean(2),axis=1)).mean()+np.abs(np.diff(v[::4,::4].astype(float).mean(2),axis=0)).mean())
            records.append(dict(scene=scene,maximum_latent_edge_adjustment=latent_error,max_channel_change_vs_r37=rgb_error,
                edge_gain_over_r37=edge(z)/edge(b),content=content(a,z),colour=colour_stats(a,z)))
        r=dict(mode=label,cap=cap,records=records,neutral_gray_exact=True,gray_curve_equal_r37=True,
            fixed_topology=True,no_postprocessing=True,training=False,laplacian_amount=.375)
        save_json(dd/'report.json',r);reports.append(r);print('BOUNDED_SHARP',json.dumps(r),flush=True)
    for scene,p in cases.items():
        a=read(p);old=rgb(prior(tensor(a)[None]*128)/128)
        board([[old,images['medium'][scene],images['strong'][scene]]],D/f'{scene}_comparison.jpg',
              ['R36 原锐度','R39 中强 / 限幅','R39 更强 / 限幅'],width=640)
    save_json(D/'summary.json',dict(reports=reports,unbounded_r38_rejected_for_halos=True,disk=disk_guard()))

if __name__=='__main__':main()
