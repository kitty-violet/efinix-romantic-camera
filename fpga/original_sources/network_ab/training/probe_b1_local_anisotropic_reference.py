"""Bounded-source image-space reference, never an FPGA/student output.

Independent implementation of the equations of Kyprianidis et al.2010:
https://www.kyprianidis.com/p/tpcg2010/jkyprian-tpcg2010.pdf
Eight polynomial sectors, RGB variance weights, structure-tensor orientation.
Only source RGB and a frozen pointwise colour map are used. No whole-region
merge or neural teacher. The existing target/manifest is never modified.
"""
import argparse,json,math,shutil,time
from pathlib import Path
import cv2,numpy as np,torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont
from .painting_parameters import tone,as_lab,as_rgb,soft_levels
from .runtime import configure_workspace,write_json,file_hash
from . import train_b1_minimum_area as m


@torch.no_grad()
def anisotropic(rgb,radius=6,power=8):
    x=torch.from_numpy(rgb.astype(np.float32)/255).permute(2,0,1).cuda()
    # Finite7x7 tensor smoothing: input dependency radius4, never global.
    z=rgb.astype(np.float32)/255
    gx=cv2.Sobel(z,cv2.CV_32F,1,0,ksize=3,scale=1/8,borderType=cv2.BORDER_REPLICATE)
    gy=cv2.Sobel(z,cv2.CV_32F,0,1,ksize=3,scale=1/8,borderType=cv2.BORDER_REPLICATE)
    st=np.stack(((gx*gx).sum(2),(gx*gy).sum(2),(gy*gy).sum(2)),0)
    st=np.stack([cv2.GaussianBlur(v,(7,7),1.2,borderType=cv2.BORDER_REPLICATE) for v in st])
    e,f,g=torch.from_numpy(st).cuda();disc=((e-g).square()+4*f.square()).sqrt()
    angle=.5*torch.atan2(2*f,e-g)+math.pi/2
    strength=(disc/(e+g).clamp_min(1e-12)).clamp(0,1)
    a=radius*(1+strength);b=radius/(1+strength);co=angle.cos();si=angle.sin()
    h,w=rgb.shape[:2];reach=2*radius;padded=F.pad(x,(reach,)*4,mode='replicate')
    sums=x.new_zeros(8,3,h,w);squares=torch.zeros_like(sums);count=x.new_zeros(8,1,h,w)
    theta=torch.arange(8,device=x.device)[:,None,None]*math.pi/4
    cc,ss=theta.cos(),theta.sin();zeta=2/radius;gamma=3*math.pi/16
    eta=(zeta+math.cos(gamma))/math.sin(gamma)**2
    for dy in range(-reach,reach+1):
      for dx in range(-reach,reach+1):
        if dx*dx+dy*dy>reach*reach:continue
        u=(dx*co+dy*si)/a;v=(-dx*si+dy*co)/b;r2=u*u+v*v
        ux=u[None]*cc+v[None]*ss;uy=-u[None]*ss+v[None]*cc
        sector=F.relu(ux+zeta-eta*uy.square()).square()
        weight=sector/sector.sum(0,keepdim=True).clamp_min(1e-20)
        weight*=torch.exp(-3.125*r2)[None]*(r2<=1)[None]
        weight=weight[:,None]
        q=padded[:,reach+dy:reach+dy+h,reach+dx:reach+dx+w][None]
        count+=weight;sums+=q*weight;squares+=q.square()*weight
    mean=sums/count.clamp_min(1e-12)
    variance=(squares/count.clamp_min(1e-12)-mean.square()).clamp_min(0).sum(1)
    # RGB shared weights; byte-scale16 controls how strongly noisy sectors lose.
    confidence=1/(1+(variance/(16/255)**2).pow(power/2))
    result=(mean*confidence[:,None]).sum(0)/confidence.sum(0).clamp_min(1e-20)
    return (result.permute(1,2,0).cpu().numpy()*255).round().clip(0,255).astype(np.uint8)


def levels(rgb,step=8):
    lab=as_lab(rgb);lab[...,0]=soft_levels(lab[...,0],step,5.,.8)
    for k in (1,2):lab[...,k]=soft_levels(lab[...,k],6.,4.,.45)
    return as_rgb(lab)


def regression():
    z=np.broadcast_to(np.array([60,112,179],np.uint8),(64,72,3)).copy()
    assert np.max(abs(anisotropic(z).astype(int)-z.astype(int)))<=1
    edge=z.copy();edge[:,36:]=[198,62,84];v=anisotropic(edge)
    assert abs(v[:,:28].astype(int)-edge[:,:28]).max()<=1
    assert abs(v[:,44:].astype(int)-edge[:,44:]).max()<=1
    changed=z.copy();changed[:12]=255;changed[-12:]=0
    assert np.array_equal(anisotropic(z)[28:36],anisotropic(changed)[28:36])
    print('LOCAL_ANISOTROPIC_PASS constant/joint_RGB_step_core/finite_support',flush=True)


def main():
    ap=argparse.ArgumentParser()
    for k in ('data','mining','adapter','out'):ap.add_argument('--'+k,required=True)
    a=ap.parse_args();configure_workspace();torch.set_num_threads(2);cv2.setNumThreads(2);regression()
    out=Path(a.out);out.mkdir(exist_ok=False);d=m.load_data(a.data);rows={r['id']:r for r in d['training']}
    specs=json.loads(Path(a.mining).read_text('utf-8'))['records'];adapter=np.load(a.adapter)['lab_chroma_affine']
    board=Image.new('RGB',(1600,1044),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',13)
    records=[];start=time.monotonic()
    for i,index in enumerate((1,5,6)):
        s=specs[index];row=rows[s['row']['id']];xx,yy=s['xy'];w,h=s['wh'];pad=32
        source=m.t.r.read(row['path']);target=m.t.r.read(row['target'])
        #32 context pixels exceed the two-pass radius24 for the320px review.
        full=np.pad(source,((pad,pad),(pad,pad),(0,0)),mode='edge')
        patch=full[yy:yy+h+2*pad,xx:xx+w+2*pad]
        coloured=tone(patch,adapter);one=anisotropic(coloured);two=anisotropic(one)
        variants=[('当前目标',target[yy:yy+h,xx:xx+w]),('局部原片',patch),
            ('8方向一次 · 仅参照',levels(one)),('8方向两次 · 仅参照',levels(two)),('两次更强色阶 · 仅参照',levels(two,11))]
        for j,(title,rgb) in enumerate(variants):
            if j:rgb=rgb[pad:-pad,pad:-pad]
            Image.fromarray(rgb).save(out/f'{row["id"]}_{j}.png');board.paste(Image.fromarray(rgb),(j*320,i*348+28))
            draw.text((j*320+3,i*348+3),title,font=font,fill='white')
        records.append(dict(id=row['id'],source_sha256=file_hash(row['path']),target_sha256=file_hash(row['target'])))
        print('LOCAL_ANISOTROPIC_RENDER',row['id'],round(time.monotonic()-start,2),flush=True)
    board.save(out/'comparison.png');shutil.copy2(__file__,out/Path(__file__).name)
    write_json(out/'results.json',dict(records=records,source_dependency_radius720=24,
        max_two_pass_support=49,constant_colour_transform=True,adapter_sha256=file_hash(a.adapter),
        inference_graph_compatibility_not_proven=True,not_B1_output=True,original_targets_modified=False,
        visual_acceptance=False,reference='https://www.kyprianidis.com/p/tpcg2010/',seconds=time.monotonic()-start))


if __name__=='__main__':main()
