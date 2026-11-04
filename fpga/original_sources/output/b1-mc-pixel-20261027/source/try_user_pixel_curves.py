"""Digitise three user-drawn curves and encode their steps in fixed B1.

Contrast120% is kept as the readout convention. The luminance basis is inverse
compensated so that the drawn red curve is the final luminance target, instead
of stretching it a second time. Chroma remains130% of the approved R3.
"""
from core import *
from train_pixel import checkpoint
from pixel_targets import panel
import r3_palette_blocks as r3
from evaluate_r3_r4 import controls

FILES=[r'C:\Users\Lenovo\AppData\Local\Temp\codex-clipboard-9940538e-5831-4add-af3d-583351b734eb.png',
       r'C:\Users\Lenovo\AppData\Local\Temp\codex-clipboard-aad55690-a97e-46dd-b1dc-377faaac3b3c.png',
       r'C:\Users\Lenovo\AppData\Local\Temp\codex-clipboard-600914e2-bd2d-4d49-a499-16e894218766.png']

def isotonic(v):
    blocks=[]
    for value in v:
        blocks.append([float(value),1])
        while len(blocks)>1 and blocks[-2][0]>blocks[-1][0]:
            b=blocks.pop();a=blocks.pop();blocks.append([(a[0]*a[1]+b[0]*b[1])/(a[1]+b[1]),a[1]+b[1]])
    return np.array([a for a,n in blocks for _ in range(n)])

def digitise(path,count):
    a=read(path);red=(a[...,0]>170)&(a[...,1]<125)&(a[...,2]<125)&(a[...,0]>a[...,1]+65)
    crop=red[133:741,91:891].astype(np.uint8);n,lab,stats,_=cv2.connectedComponentsWithStats(crop,connectivity=8)
    # A hand-drawn curve may contain pen lifts. Keep every substantial stroke,
    # rather than silently extending only its largest connected component.
    keep=[k for k in range(1,n) if stats[k,cv2.CC_STAT_AREA]>=20 and
          max(stats[k,cv2.CC_STAT_WIDTH],stats[k,cv2.CC_STAT_HEIGHT])>=12]
    mask=np.isin(lab,keep)
    y=np.full(256,np.nan)
    for i in range(256):
        x=round(i/255*(881-91));ys=np.where(mask[:,max(0,x-1):min(mask.shape[1],x+2)])[0]
        if len(ys):y[i]=(731-(np.median(ys)+133))/(731-152)*255
    good=np.where(np.isfinite(y))[0];y=np.interp(np.arange(256),good,y[good]);y=isotonic(np.clip(y,0,255))
    # Piecewise-constant least squares, with explicit maximum channel budget.
    n=256;cs=np.r_[0,y.cumsum()];sq=np.r_[0,(y*y).cumsum()]
    dp=np.full((count+1,n+1),np.inf);back=np.zeros((count+1,n+1),int);dp[0,0]=0
    for k in range(1,count+1):
        for j in range(k,n+1):
            starts=np.arange(k-1,j);length=j-starts
            cost=dp[k-1,starts]+sq[j]-sq[starts]-(cs[j]-cs[starts])**2/length
            z=int(cost.argmin());dp[k,j]=cost[z];back[k,j]=starts[z]
    bounds=[];j=n
    for k in range(count,0,-1):i=int(back[k,j]);bounds.append((i,j));j=i
    bounds=bounds[::-1];levels=np.array([y[i:j].mean() for i,j in bounds]);levels[0]=0;levels[-1]=255
    steps=np.concatenate([np.full(j-i,v) for (i,j),v in zip(bounds,levels)])
    return dict(bounds=[list(x) for x in bounds],thresholds=[i-.5 for i,j in bounds[1:]],levels=levels.tolist(),
        digitised_monotone=y.tolist(),step_fit=steps.tolist(),step_fit_rmse=float(np.sqrt(np.mean((steps-y)**2))),
        source_file=str(path),source_sha256=sha(path),plot_coordinates=dict(x0=91,x255=881,y0=731,y255=152),
        interpretation='monotone major steps; ignores handwritten overshoot, tiny backtracking, isolated red dots')

def requantise(n):
    for key,w in n.weight.items():
        need=(w.abs().flatten(1).amax(1)*131072/127).clamp_min(64)
        getattr(n,key+'_m').copy_((2**need.log2().ceil()).clamp(64,131071).double()[None,:,None,None])

def scalar_head_curve(n):
    params,_=decode(n.export());p=params['rgb_head'];values=[]
    for z in range(128):
        x=[z,64,64,0,0,0,0,0];out=[]
        for w,b,m in zip(p.weights,p.bias,p.multiplier):
            acc=max(-(2**31),min(2**31-1,sum(int(a)*int(v) for a,v in zip(w,x))+b));product=acc*m
            q=(abs(product)+65536)//131072*(1 if product>=0 else -1);out.append(max(-128,min(127,q))+128)
        values.append(np.mean(out))
    return np.array(values)

@torch.no_grad()
def seed(c):
    count=len(c['levels']);assert count-1+16<=32
    n=r3.seed(dict(morph='OC3',blur=True,face=.05,levels_y=count,levels_c=9))
    for j,t in enumerate(c['thresholds']):
        n.weight['down2__pw'][j,:3,0,0]=torch.tensor([.25,.5,.25],device='cuda')*126
        n.bias['down2__pw'][j]=float(63.5-126*t/2)
    contrast=1.2;saturation=1.3;w=n.weight['rgb_head'];b=n.bias['rgb_head'];centre=w[:,1:3,0,0].sum(1)*63.5
    for col in [0,3,4]:w[:,col].mul_(contrast)
    w[:,1:3].mul_(saturation);b.sub_(128*(contrast-1)+(saturation-1)*centre);requantise(n)
    headcurve=scalar_head_curve(n);latent=[int(np.abs(headcurve-v).argmin()) for v in c['levels']]
    # Recover each desired plateau with ordinary INT8 basis weights. m=256
    # permits quarter-latent-unit increments per integer mask coefficient.
    wi=[];acc=latent[0]*512;bias=acc
    for z in latent[1:]:
        q=round((z*512-acc)/127);assert 0<=q<=127,(z,q,acc,latent,c['levels'])
        wi.append(q);acc+=127*q
    for row in [0,3,4]:
        n.weight['res3_conv2__pw'][row].zero_();n.bias['res3_conv2__pw'][row]=bias/512
        n.weight['res3_conv2__pw'][row,:count-1,0,0]=torch.tensor(wi,device='cuda')/512
        n.res3_conv2__pw_m[:,row]=256
    return n,latent

def teacher(a,c):
    z=cv2.GaussianBlur(a.astype(np.float32),(3,3),0,borderType=cv2.BORDER_REPLICATE);h,w=z.shape[:2]
    r,g,b=z.reshape(h//4,4,w//4,4,3).mean((1,3)).transpose(2,0,1)
    y=(r+2*g+b)/4;Y=np.full_like(y,c['levels'][0])
    for t,inc in zip(c['thresholds'],np.diff(c['levels'])):
        m=(y>t).astype(np.uint8)
        for op in 'EDDIE':
            if op=='D':m=cv2.dilate(m,np.ones((3,3),np.uint8),borderType=cv2.BORDER_REPLICATE)
            elif op=='E':m=cv2.erode(m,np.ones((3,3),np.uint8),borderType=cv2.BORDER_REPLICATE)
        Y+=inc*m
    co=(r-b)/2;cg=(-r+2*g-b)/4
    def chroma(v):
        q=np.full_like(v,r3.CHROMA_VALUES[0])
        for t,inc in zip(r3.CHROMA_THRESH,np.diff(r3.CHROMA_VALUES)):q+=inc*(v>t)
        return q*1.3
    co,cg=chroma(co),chroma(cg);top=np.pad(Y,((1,0),(0,0)),mode='edge')[:-1];right=np.pad(Y,((0,0),(0,1)),mode='edge')[:,1:]
    shade=.05*(np.maximum(Y-top,0)-1.2*np.maximum(Y-right,0));Y=Y+shade
    return np.clip(np.stack([Y+co-cg,Y+cg,Y-co-cg],2)+.5,0,255).astype(np.uint8).repeat(4,0).repeat(4,1)

def plot_curves(curves,actual,path):
    im=Image.new('RGB',(1330,845),'#f8faff');draw=ImageDraw.Draw(im);font=lambda s:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',s)
    draw.text((45,20),'三条手绘曲线：提取目标与实际 B1 整数响应',font=font(28),fill='#20354e')
    left,top,w,h=95,135,785,590;xy=lambda x,y:(int(left+x/255*w),int(top+(255-y)/255*h))
    for v in [0,32,64,96,128,160,192,224,255]:
        x,y=xy(v,v);draw.line((x,top,x,top+h),fill='#dce5f0');draw.line((left,y,left+w,y),fill='#dce5f0')
        draw.text((x-13,top+h+10),str(v),font=font(16),fill='#46536a');draw.text((left-45,y-9),str(v),font=font(16),fill='#46536a')
    colors=['#cc3333','#1878bf','#2d9b54']
    for j,(c,v,color) in enumerate(zip(curves,actual,colors)):
        light='#'+''.join(f'{(int(color[k:k+2],16)+255)//2:02x}' for k in [1,3,5])
        for start in range(0,253,5):
            draw.line([xy(i,c['digitised_monotone'][i]) for i in range(start,min(start+3,256))],fill=light,width=2)
        draw.line([xy(i,z) for i,z in enumerate(v)],fill=color,width=4)
        draw.text((925,160+j*165),f"{'ABC'[j]}：{len(c['levels'])}档亮度",font=font(23),fill=color)
        draw.multiline_text((925,205+j*165),'按手绘红线锁定最终亮度\n保留阶梯 / 饱和度130%\n固定B1系数，无额外滤镜',font=font(18),fill='#52647d',spacing=12)
    draw.text((260,785),'输入灰度 → 实际 B1 平均 RGB 灰度',font=font(21),fill='#20354e')
    draw.text((95,815),'浅虚线：手绘提取目标；深实线：实际整数输出。手绘回勾/超界按单调主平台解释。',font=font(16),fill='#52647d');im.save(path)

def main():
    setup(271081);root=OUT/'pixel-three-user-curves';root.mkdir(exist_ok=True);disk_guard();curves=[];actual=[]
    for letter,path,count in zip('ABC',FILES,[9,16,15]):
        d=root/letter;d.mkdir(exist_ok=True);c=digitise(path,count);n,latent=seed(c);raw=n.export()
        (d/'candidate.bin').write_bytes(raw);checkpoint(n,d/'selected.pt',0,trained=False,curve=c,contrast=1.2,saturation=1.3,
            kind='user drawn final luminance curve with inverse contrast compensation')
        eng=IntegerEngine(raw);gray=controls(dict(candidate=eng))['candidate'];actual.append(gray['gray_curve_mean']);curves.append(c)
        records=[]
        for scene in ['outdoor','indoor']:
            a=read(OUT/f'targets/{scene}_source.png');t=teacher(a,c);o=rgb(eng(tensor(a)[None]*128)/128)
            assert torch.equal(n(tensor(a)[None])*128,eng(tensor(a)[None]*128))
            Image.fromarray(o).save(d/f'{scene}_student.png');Image.fromarray(t).save(d/f'{scene}_target.png')
            panel([a,t,o],['原图',f'手绘曲线{letter}目标 / S130',f'实际B1 曲线{letter}'],d/f'{scene}_comparison.jpg')
            records.append(dict(scene=scene,target_mae=float(np.abs(t.astype(float)-o).mean())))
        save_json(d/'curve.json',dict(**c,latent_levels=latent,actual=gray,static=records,coefficient_sha256=sha(d/'candidate.bin'),
            channel_budget=count-1+16,contrast=1.2,saturation=1.3,contrast_inverse_compensated=True,final_luminance_matches_drawn_curve=True,
            actual_curve_vs_drawn_rmse=float(np.sqrt(np.mean((np.array(actual[-1])-np.array(c['digitised_monotone']))**2))),
            added_inference_operators=False,trained=False,temporal_verified=False))
        print('USER_CURVE_BUILT',letter,'LEVELS',count,'RMSE',round(float(np.sqrt(np.mean((np.array(actual[-1])-np.array(c['digitised_monotone']))**2))),3),
            'TABLE',json.dumps(list(zip(c['bounds'],[round(v,1) for v in c['levels']]))),flush=True)
    plot_curves(curves,actual,root/'actual-three-curves.png')
    print('THREE_USER_CURVES_COMPLETE',flush=True)

if __name__=='__main__':main()
