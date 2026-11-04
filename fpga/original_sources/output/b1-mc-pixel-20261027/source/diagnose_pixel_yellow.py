"""Measure colour bias on neutral/warm inputs and paired real frames."""
from core import *
from evaluate_r3_r4 import controls

D=OUT/'pixel-yellow-diagnosis';D.mkdir(exist_ok=True)

@torch.inference_mode()
def main():
    setup();disk_guard()
    paths={'original_A':OUT/'exports/pixel-user-curve-A/coefficients-padded.bin',
           'r36':OUT/'exports/pixel-A-crisp-r36/coefficients-padded.bin'}
    engines={k:IntegerEngine(p.read_bytes()) for k,p in paths.items()}
    samples=[[128+d,128,128-d] for d in range(-20,21)]+[[160,160,160],[160,158,144],[200,198,184],[170,165,153],[130,150,170],[80,140,220],[220,160,40]]
    a=torch.tensor(samples,device='cuda',dtype=torch.float32)[:,:,None,None].expand(-1,-1,16,16)-128
    outputs={k:(e(a)[:,:,8,8]+128).cpu().numpy().astype(int).tolist() for k,e in engines.items()}
    swatches=[dict(input=c,**{k:v[i] for k,v in outputs.items()}) for i,c in enumerate(samples)]
    scenes={s:read(p) for s,p in {'corridor':OUT/'data-quality/VID_20260924_231158_source.png','outdoor':OUT/'targets/outdoor_source.png'}.items()}
    rows=[]
    for scene,source in scenes.items():
        images={'source':source,**{k:rgb(e(tensor(source)[None]*128)/128) for k,e in engines.items()}}
        row={'scene':scene,'images':{}}
        z=source.astype(float);spread=z.max(2)-z.min(2);light=(z.mean(2)>40)&(z.mean(2)<230)
        weak=(spread>=12)&(spread<=28)&light
        neutral=(spread<=8)&light
        for name,v in images.items():
            vv=v.astype(float);yellow=(vv[:,:,0]+vv[:,:,1])/2-vv[:,:,2]
            row['images'][name]=dict(mean_yellow=float(yellow.mean()),
                weak_chroma_mean_spread=float((vv.max(2)-vv.min(2))[weak].mean()),
                nearneutral_mean_yellow=float(yellow[neutral].mean()),
                nearneutral_yellow_fraction=float((yellow[neutral]>18).mean()))
        yellow=(images['r36'][...,0].astype(float)+images['r36'][...,1].astype(float))/2-images['r36'][...,2]
        ys,xs=np.where(weak&(yellow>22))
        order=np.argsort(yellow[ys,xs])
        coords=[]
        for j in order[::-1]:
            y,x=int(ys[j]),int(xs[j])
            if all((x-xx)**2+(y-yy)**2>120**2 for yy,xx in coords):coords.append((y,x))
            if len(coords)>=6:break
        row['examples']=[dict(x=x,y=y,rgb={k:v[y,x].tolist() for k,v in images.items()}) for y,x in coords]
        rows.append(row)
        board([[images[k] for k in images]],D/f'{scene}_comparison.jpg',['原图','原曲线A','R36'],width=640)
    n=load_model(OUT/'exports/pixel-A-crisp-r36/selected.pt')
    params,_=decode(n.export())
    detail={k:dict(weights=params[k].weights,bias=params[k].bias,multiplier=params[k].multiplier) for k in ['down2.pw','res3_conv2.pw','rgb_head']}
    report=dict(swatches=swatches,scenes=rows,controls=controls(engines),coefficients=detail)
    save_json(D/'diagnosis.json',report)
    print(json.dumps(dict(swatches=swatches[18:35]+swatches[41:],scenes=rows),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
