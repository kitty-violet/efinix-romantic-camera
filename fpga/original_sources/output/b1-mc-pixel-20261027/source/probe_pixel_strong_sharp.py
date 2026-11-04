"""Visible pixel-boundary contrast using the existing low-resolution Y depthwise."""
from core import *
from evaluate_r3_r4 import controls,content
from probe_pixel_colour_sharp import colour_stats

D=OUT/'pixel-r38-strong-sharp'
BASE=OUT/'exports/pixel-A-colour-sharp-r37/selected.pt'

@torch.no_grad()
def make(amount):
    n=load_model(BASE);k='up1_low_dw'
    kernel=torch.zeros((3,3),device='cuda');kernel[1,1]=1+4*amount
    for y,x in [(0,1),(1,0),(1,2),(2,1)]:kernel[y,x]=-amount
    n.weight[k][0,0]=kernel;n.bias[k][0]=0
    # All fractions chosen exactly representable; uniform grey DC gain stays 1.
    scale=1/64 if amount<=.1875 else 1/32
    getattr(n,k+'_m')[:,0]=int(scale*131072)
    q=kernel/scale;assert q.abs().max()<=127 and torch.equal(q,q.round()) and float(q.sum())*scale==1
    return n

@torch.inference_mode()
def main():
    setup();D.mkdir(exist_ok=True);disk_guard();shutil.copy2(__file__,D/'source_used.py')
    prior=IntegerEngine((OUT/'exports/pixel-A-crisp-r36/coefficients-padded.bin').read_bytes())
    mild=IntegerEngine((OUT/'exports/pixel-A-colour-sharp-r37/coefficients-padded.bin').read_bytes())
    curves=controls({'prior':prior,'mild':mild});reports=[];images={}
    cases={'corridor':OUT/'data-quality/VID_20260924_231158_source.png','outdoor':OUT/'targets/outdoor_source.png',
           'indoor_training':OUT/'data-quality/VID_20260925_070939_source.png'}
    for label,amount in [('medium',.1875),('strong',.375)]:
        dd=D/label;dd.mkdir(exist_ok=True);n=make(amount);raw=n.export();e=IntegerEngine(raw)
        (dd/'candidate.bin').write_bytes(raw)
        torch.save(dict(model=n.state_dict(),residual_encoding=copy.deepcopy(n.package['residual_encoding']),
            qat=True,step=0,trained=False,sharpen_amount=amount,source=str(BASE)),dd/'constructed.pt')
        ctrl=controls({'e':e})['e'];assert ctrl['gray_sweep_max_channel_difference']==0
        assert np.array_equal(ctrl['gray_curve_mean'],curves['mild']['gray_curve_mean'])
        rows=[];images[label]={}
        for scene,p in cases.items():
            a=read(p);x=tensor(a)[None]*128;nodes,_=e(x,True);ref,_=mild(x,True)
            assert torch.equal(nodes['res3_conv2.pw'],ref['res3_conv2.pw'])
            assert torch.equal(n(x/128)*128,nodes['rgb_head'])
            z=rgb(nodes['rgb_head']/128);b=rgb(ref['rgb_head']/128);old=rgb(prior(x)/128)
            Image.fromarray(z).save(dd/f'{scene}.png');images[label][scene]=z
            low=z[::4,::4].astype(float);bl=b[::4,::4].astype(float)
            edge=lambda v:float(np.abs(np.diff(v.mean(2),axis=1)).mean()+np.abs(np.diff(v.mean(2),axis=0)).mean())
            flat=z.reshape(180,4,320,4,3);assert np.all(flat.max((1,3))==flat.min((1,3)))
            rows.append(dict(scene=scene,edge_contrast=edge(low),mild_edge_contrast=edge(bl),
                edge_gain_over_mild=edge(low)/edge(bl),rgb_mae_vs_mild=float(np.abs(z.astype(float)-b).mean()),
                content=content(a,z),colour=colour_stats(a,z),colour_before=colour_stats(a,old)))
        r=dict(mode=label,amount=amount,neutral_gray_exact=True,gray_curve_equal_r37=True,
            original_Y_palette_unchanged=True,edge_contrast_intentionally_changed=True,records=rows,
            effects='Y-only symmetric edge emphasis before existing upsampling; hue coefficients unchanged; no postfilter')
        save_json(dd/'report.json',r);reports.append(r);print('STRONG_SHARP',json.dumps(r),flush=True)
    for scene,p in cases.items():
        a=read(p);old=rgb(prior(tensor(a)[None]*128)/128)
        board([[old,images['medium'][scene],images['strong'][scene]]],D/f'{scene}_comparison.jpg',
              ['R36 原锐度','R38 中强 + 修正偏黄','R38 更强 + 修正偏黄'],width=640)
        yy,xx=(0,180) if scene=='corridor' else (160,640)
        crop=[v[yy:yy+220,xx:xx+300] for v in [old,images['medium'][scene],images['strong'][scene]]]
        im=Image.new('RGB',(1800,470),'#171d26');dr=ImageDraw.Draw(im);f=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22)
        for i,(z,label) in enumerate(zip(crop,['R36 原锐度','R38 中强','R38 更强'])):
            im.paste(Image.fromarray(z).resize((600,440),Image.Resampling.NEAREST),(i*600,30));dr.text((i*600+8,2),label,font=f,fill='white')
        im.save(D/f'{scene}_crop.png')
    save_json(D/'summary.json',dict(reports=reports,earlier_r37_video_stopped_by_new_user_request=True,disk=disk_guard()))

if __name__=='__main__':main()
