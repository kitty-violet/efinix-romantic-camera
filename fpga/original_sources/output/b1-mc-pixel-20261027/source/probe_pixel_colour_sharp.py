"""Neutral integer centring, smaller first chroma steps and bounded stem detail."""
from core import *
from evaluate_r3_r4 import controls,content
from probe_pixel_crisp_edges import stats

D=OUT/'pixel-r37-colour-sharp'
BASE=OUT/'exports/pixel-A-crisp-r36/selected.pt'

def away(v):return int(np.sign(v)*np.floor(abs(v)+.5))

@torch.no_grad()
def make(low_chroma=None,detail=0.):
    n=load_model(BASE);params,_=decode(n.export())
    p=params['rgb_head'];w=np.array(p.weights);b=np.array(p.bias)
    effective=b+(w[:,1]+w[:,2])*64
    common=int(np.rint(effective.mean()))
    # Exactly equal accumulator on integer-neutral Co=Cg=64, all luminance codes.
    neutral_bias=common-(w[:,1]+w[:,2])*64
    scales=np.array(p.multiplier)/131072
    n.bias['rgb_head'].copy_(torch.tensor(neutral_bias*scales,device='cuda',dtype=torch.float32))
    palette_info=[]
    if low_chroma is not None:
        key='res3_conv2__pw';p=params['res3_conv2.pw']
        for row,start in [(1,8),(2,16)]:
            weights=np.array(p.weights[row]);acc=int(p.bias[row]);m=p.multiplier[row]
            old=[away(acc*m/131072)]
            for v in weights[start:start+8]:
                acc+=int(v)*127;old.append(away(acc*m/131072))
            wanted=old.copy();wanted[3]=64-low_chroma//2;wanted[5]=64+low_chroma//2
            acc=old[0]*512;newbias=acc;coeff=[]
            for value in wanted[1:]:
                wi=round((value*512-acc)/127);coeff.append(wi);acc+=wi*127
            assert all(0<=v<=127 for v in coeff)
            n.weight[key][row].zero_();n.weight[key][row,start:start+8,0,0]=torch.tensor(coeff,device='cuda')/512
            n.bias[key][row]=newbias/512;getattr(n,key+'_m')[:,row]=256
            palette_info.append(dict(row=row,old_latent=old,new_latent=wanted))
    if detail:
        g=np.outer([1,2,1],[1,2,1])/16;eye=np.zeros((3,3));eye[1,1]=1
        kernel=.5*((1-detail)*g+detail*eye)
        # detail=1 removes only the prior small Gaussian; 1.25 adds mild unsharp.
        for c in range(3):n.weight['stem'][c,c].copy_(torch.tensor(kernel,device='cuda'))
        need=(n.weight['stem'][:3].abs().flatten(1).amax(1)*131072/127).clamp_min(64)
        n.stem_m[:,:3].copy_((2**need.log2().ceil()).clamp(64,131071).double()[None,:,None,None])
    return n,dict(low_chroma=low_chroma,detail=detail,neutral_accumulator=common,palette=palette_info,
        source='pixel-A-crisp-r36',saturation_gain_preserved=True,training=False)

def colour_stats(a,z):
    # Match the 4x4 prefilter support for a fair colour rather than geometry comparison.
    f=cv2.GaussianBlur(a.astype(np.float32),(3,3),0,borderType=cv2.BORDER_REPLICATE)
    f=f.reshape(180,4,320,4,3).mean((1,3)).repeat(4,0).repeat(4,1)
    spread=f.max(2)-f.min(2);weak=(spread>10)&(spread<32)&(f.mean(2)>40)&(f.mean(2)<230)
    zz=z.astype(float);outspread=zz.max(2)-zz.min(2)
    excess=np.maximum(outspread-1.3*spread,0)
    return dict(weak_colour_excess=float(excess[weak].mean()),weak_output_spread=float(outspread[weak].mean()),
                mean_yellow=float(((zz[:,:,0]+zz[:,:,1])/2-zz[:,:,2]).mean()))

@torch.inference_mode()
def main():
    setup();D.mkdir(exist_ok=True);disk_guard();shutil.copy2(__file__,D/'source_used.py')
    prior=IntegerEngine((OUT/'exports/pixel-A-crisp-r36/coefficients-padded.bin').read_bytes())
    configs=dict(neutral=(None,0),colour12=(12,0),colour8=(8,0),colour12_detail=(12,1),colour12_sharp=(12,1.25))
    cases={'corridor':OUT/'data-quality/VID_20260924_231158_source.png','outdoor':OUT/'targets/outdoor_source.png',
           'indoor_training':OUT/'data-quality/VID_20260925_070939_source.png'}
    gray=controls({'base':prior})['base']['gray_curve_mean']
    images={};reports=[]
    probes=[[128,128,128],[137,128,119],[145,128,111],[100,128,156],[80,140,220],[220,160,40]]
    patch=torch.tensor(probes,device='cuda',dtype=torch.float32)[:,:,None,None].expand(-1,-1,16,16)-128
    for mode,(chroma,detail) in configs.items():
        d=D/mode;d.mkdir(exist_ok=True);n,config=make(chroma,detail);raw=n.export();e=IntegerEngine(raw)
        (d/'candidate.bin').write_bytes(raw)
        torch.save(dict(model=n.state_dict(),residual_encoding=copy.deepcopy(n.package['residual_encoding']),
            qat=True,step=0,trained=False,config=config),d/'constructed.pt')
        ctrl=controls({'new':e})['new'];assert ctrl['gray_sweep_max_channel_difference']==0
        delta=float(np.max(np.abs(np.array(ctrl['gray_curve_mean'])-gray)));assert delta<=1
        rows=[];images[mode]={}
        for scene,p in cases.items():
            a=read(p);x=tensor(a)[None]*128;nodes,_=e(x,True);old,_=prior(x,True)
            if not detail:
                assert torch.equal(nodes['res3_conv2.dw'],old['res3_conv2.dw'])
                assert torch.equal(nodes['res3_conv2.pw'][:,0],old['res3_conv2.pw'][:,0])
            assert torch.equal(n(x/128)*128,nodes['rgb_head'])
            z=rgb(nodes['rgb_head']/128);b=rgb(old['rgb_head']/128);images[mode][scene]=z
            Image.fromarray(z).save(d/f'{scene}.png')
            rows.append(dict(scene=scene,colour=colour_stats(a,z),prior_colour=colour_stats(a,b),
                stats=stats(z),content=content(a,z),prior_content=content(a,b),rgb_mae=float(np.abs(z.astype(float)-b).mean())))
        po=(e(patch)[:,:,8,8]+128).cpu().numpy().astype(int).tolist()
        result=dict(mode=mode,config=config,gray_max_luma_change=delta,neutral_gray_exact=True,
                    controls=ctrl,swatches=[dict(input=a,output=b) for a,b in zip(probes,po)],records=rows)
        save_json(d/'report.json',result);reports.append(result)
        print('COLOUR_SHARP',mode,json.dumps(dict(gray_delta=delta,swatches=result['swatches'],records=rows)),flush=True)
    for scene,p in cases.items():
        source=read(p);old=rgb(prior(tensor(source)[None]*128)/128)
        shown=[source,old,images['colour12'][scene],images['colour12_sharp'][scene]]
        board([shown],D/f'{scene}_comparison.jpg',['原图','R36 / 修正前','中性点与浅色修正','浅色修正 + 适度锐化'],width=640)
        y,x=(0,120) if scene=='corridor' else (160,680)
        crops=[v[y:y+224,x:x+280] for v in shown]
        im=Image.new('RGB',(1120,948),'#171d26');draw=ImageDraw.Draw(im);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
        for i,(a,label) in enumerate(zip(crops,['原图','R36','颜色修正','颜色修正 + 锐化'])):
            xx=(i%2)*560;yy=(i//2)*474
            im.paste(Image.fromarray(a).resize((560,448),Image.Resampling.NEAREST),(xx,yy+26));draw.text((xx+8,yy),label,font=font,fill='white')
        im.save(D/f'{scene}_crop.png')
    save_json(D/'summary.json',dict(reports=reports,disk=disk_guard(),MC_closed=True))

if __name__=='__main__':main()
