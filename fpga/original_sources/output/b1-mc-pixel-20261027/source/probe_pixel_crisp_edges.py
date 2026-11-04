"""Remove edge-only extra shades without changing threshold-mask membership.

All operations are encoded in the existing B1 DW and RGB-head coefficients.
No inference postprocessing, no new pixel-grid phase, no region dilation.
"""
from core import *
from evaluate_r3_r4 import controls,content
from r13_region_cuboids import region_stats

D=OUT/'pixel-r36-crisp-edges'

@torch.no_grad()
def make(mode):
    n=load_model(OUT/'exports/pixel-A-fringe-r28/selected.pt')
    if 'hard' in mode:
        k='res3_conv2__dw';s=131071/131072
        n.weight[k][:24].zero_();n.weight[k][:24,0,1,1]=127*s
        n.bias[k][:24]=-63*127*s
        getattr(n,k+'_m')[:,:24].fill_(131071)
    if 'flat' in mode:n.weight['rgb_head'][:,3:5].zero_()
    return n

def stats(z):
    low=z.reshape(180,4,320,4,3)
    within=np.max(low,axis=(1,3))-np.min(low,axis=(1,3))
    unique=len(np.unique(z.reshape(-1,3),axis=0))
    return dict(rgb_colours=unique,nonconstant_4x4_fraction=float(np.any(within>0,axis=2).mean()),
                regions=region_stats(z))

@torch.inference_mode()
def main():
    setup(271200);D.mkdir(exist_ok=True);disk_guard();shutil.copy2(__file__,D/'source_used.py')
    original=IntegerEngine((OUT/'exports/pixel-user-curve-A/coefficients-padded.bin').read_bytes())
    previous=IntegerEngine((OUT/'exports/pixel-A-fringe-r28/coefficients-padded.bin').read_bytes())
    gray=controls({'A':original})['A']['gray_curve_mean']
    cases={'corridor':OUT/'data-quality/VID_20260924_231158_source.png',
           'outdoor':OUT/'targets/outdoor_source.png',
           'indoor_training':OUT/'data-quality/VID_20260925_070939_source.png'}
    images={};reports=[]
    for name in ['r28','hard','flat','hard-flat']:
        dd=D/name;dd.mkdir(exist_ok=True);n=make(name);raw=n.export();eng=IntegerEngine(raw)
        torch.save(dict(model=n.state_dict(),residual_encoding=copy.deepcopy(n.package['residual_encoding']),
            qat=True,step=0,trained=False,construction=name,source='pixel-A-fringe-r28',
            purpose='pure pixel layer clean-up; preserve membership and curve A'),dd/'constructed.pt')
        (dd/'candidate.bin').write_bytes(raw)
        g=controls({'candidate':eng})['candidate']['gray_curve_mean']
        assert np.array_equal(g,gray),(name,'A curve changed')
        rows=[];images[name]={}
        for scene,p in cases.items():
            a=read(p);x=tensor(a)[None]*128
            nodes,_=eng(x,True);prior,_=previous(x,True)
            assert torch.equal(nodes['res3_conv1.pw'],prior['res3_conv1.pw'])
            before=prior['res3_conv2.dw'][:,:24];after=nodes['res3_conv2.dw'][:,:24]
            # Existing whole-region geometry is unchanged at its explicit binary membership threshold.
            assert torch.equal(before>=64,after>=64),(scene,name,'region membership changed')
            if 'hard' in name:assert bool(((after==0)|(after==127)).all())
            out=rgb(nodes['rgb_head']/128);base=rgb(prior['rgb_head']/128)
            assert torch.equal(n(x/128)*128,nodes['rgb_head'])
            Image.fromarray(out).save(dd/f'{scene}.png');images[name][scene]=out
            delta=np.abs(out.astype(float)-base)
            soft=(before>0)&(before<127)
            rows.append(dict(scene=scene,prior_soft_mask_fraction=float(soft.float().mean()),
                soft_Y_fraction=float(soft[:,:8].float().mean()),soft_chroma_fraction=float(soft[:,8:].float().mean()),
                exact_threshold_membership_preserved=True,raw_delta_rgb_mae=float(delta.mean()),
                changed_pixel_fraction=float(np.any(delta>0,axis=2).mean()),
                current=stats(out),prior=stats(base),content=content(a,out)))
        report=dict(variant=name,gray_curve_A_exact=True,records=rows,trained=False,
            ordinary_B1=True,coefficient_sha256=sha(dd/'candidate.bin'),
            note='Membership-preserving does not mean every RGB value is unchanged. Mid-edge levels and optional directional shades are intentionally removed.')
        save_json(dd/'report.json',report);reports.append(report)
        print('PIXEL_CRISP',name,json.dumps(rows,ensure_ascii=False),flush=True)
    for scene,p in cases.items():
        a=read(p);old=rgb(original(tensor(a)[None]*128)/128)
        board([[old,images['r28'][scene],images['hard-flat'][scene]]],D/f'{scene}_comparison.jpg',
              ['原曲线A','R28：去孤立杂色','R36：硬色层、去边缘额外明暗'],width=640)
        # Native pixels shown enlarged without smoothing.
        yy,xx=(80,280) if scene=='corridor' else ((180,560) if scene=='outdoor' else (40,640))
        crops=[old[yy:yy+192,xx:xx+256],images['r28'][scene][yy:yy+192,xx:xx+256],images['hard-flat'][scene][yy:yy+192,xx:xx+256]]
        canvas=Image.new('RGB',(1536,420),'#171b22');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
        for j,(label,c) in enumerate(zip(['原曲线A','R28','R36候选'],crops)):
            canvas.paste(Image.fromarray(c).resize((512,384),Image.Resampling.NEAREST),(512*j,36));draw.text((512*j+8,5),label,font=font,fill='white')
        canvas.save(D/f'{scene}_crop.png')
    save_json(D/'summary.json',dict(reports=reports,MC_route_closed=True,curve_A_preserved=True,selected_by_user=False,disk=disk_guard()))
    print('PIXEL_CRISP_COMPLETE',flush=True)

if __name__=='__main__':main()
