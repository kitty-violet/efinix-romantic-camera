"""Ablate existing coefficient routes; no new operators or inference filters."""
import copy
from PIL import Image,ImageDraw,ImageFont
from common import *

@torch.no_grad()
def candidate(kind):
    m=model()
    if kind.startswith('gate'):
        corners,edges,centre={'gate025':(2,4,103),'gate050':(4,8,79),'gate100':(8,16,31)}[kind]
        q=torch.tensor([[corners,edges,corners],[edges,centre,edges],[corners,edges,corners]],device='cuda').float()
        sc=m.up1_low_dw_m.float()/131072
        for c in range(24):m.w['up1_low_dw'][c,0]=q*sc[c]
    if kind in ['ring','ring_bias','bias']:
        if 'ring' in kind:
            for n in ['res1_a_dw','res1_b_dw']:
                s=(getattr(m,n+'_m').float()/131072)
                qi=o.HalfAway.apply(m.w[n]/s[:,None,None,None]).clamp(-127,127)
                for c in range(24):
                    total=int(qi[c].sum());edge=min(127,total//8);centre=total-8*edge
                    assert 0<=centre<=127
                    qi[c,0].fill_(edge);qi[c,0,1,1]=centre
                m.w[n].copy_(qi*s[:,None,None,None])
        if 'bias' in kind:
            _,ps=o.decode(R2,2);p=ps['res2_b_pw'];s=p['multiplier'].astype(float)/131072
            m.b['res1_b_pw'].copy_(torch.tensor(p['bias']*s,device='cuda',dtype=torch.float32))
    return m

@torch.inference_mode()
def main():
    o.setup(20261102);guard();(D/'diagnosis').mkdir(exist_ok=True)
    names=['r1_before','r2','ring','bias','ring_bias','gate025','gate050','gate100']
    ms={'r1_before':o.Engine(R1),'r2':o.Engine(R2,2)}
    ms.update({n:candidate(n) for n in names[2:]})
    cases={p.stem:p for p in (D/'inputs').glob('*.png')}
    original=o.r2.OLD/'acceptance-four-scenes-5s-20261029/originals'
    cases.update({n:original/f'{n}_076.png' for n in ['indoor','outdoor','crowd','water']})
    report=[]
    for case,path in cases.items():
        a=np.array(Image.open(path).convert('RGB'));x=o.tx(a[None]);out={n:rgb(m(x)) for n,m in ms.items()}
        for n,b in out.items():
            row=dict(case=case,variant=n,**block_metrics(b,a),mae_r2=float(np.abs(b.astype(float)-out['r2']).mean()))
            report.append(row);Image.fromarray(b).save(D/'diagnosis'/f'{case}_{n}.png')
        print('PIXEL_DIAG',case,[(r['variant'],r['isolated_cells'],r['unsupported_jumps'],round(r['mae_r2'],2),r['unique_colours']) for r in report if r['case']==case],flush=True)
    save(D/'diagnosis/metrics.json',report)
    # Uniform response and colour basis are checked directly from pixels.
    xs=np.broadcast_to(np.arange(256,dtype=np.uint8)[None,:,None],(32,256,3)).copy()
    curves={n:rgb(m(o.tx(xs[None])))[16,:,0].tolist() for n,m in ms.items()}
    save(D/'diagnosis/gray-curves.json',curves)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
    panel=Image.new('RGB',(2560,2256),'#101820');dr=ImageDraw.Draw(panel)
    for i,n in enumerate(['r1_before','r2','ring','ring_bias','gate025','gate050']):
        x=i%2*1280;y=i//2*752;dr.text((x+8,y+4),n,font=font,fill='white');panel.paste(Image.open(D/'diagnosis'/f'photo_{n}.png'),(x,y+32))
    panel.save(D/'diagnosis/photo_ablation.jpg',quality=95)
    print('DIAG_COMPLETE',flush=True)

if __name__=='__main__':main()
