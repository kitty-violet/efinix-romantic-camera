"""Training targets sharing one geometric interface for all colour predicates.

Existing targets measure distance independently to30 nested colour unions.
Near a multi-colour junction these unions can select different nearest edges.
This diagnostic instead measures the nearest palette interface once, then
uses its two colour codes for all30 fields. B1 inference is unchanged.
"""
import cv2,numpy as np,torch
from torch.nn import functional as F
from .build_b1_thirty_binary_regions import code_bits,GROUPS30


def joint_interface_fields(code,gain=8,with_stats=False):
    result=[];records=[]
    for own in code.detach().cpu().numpy().astype(np.int32):
        h,w=own.shape;shape=(2*h+1,2*w+1)
        seam=np.zeros(shape,bool);ca=np.zeros(shape,np.int32);cb=np.zeros(shape,np.int32)
        yy,xx=np.nonzero(own[:,1:]!=own[:,:-1])
        for d in (-1,0,1):
            y=2*yy+1+d;x=2*xx+2;seam[y,x]=True;ca[y,x]=own[yy,xx];cb[y,x]=own[yy,xx+1]
        yy,xx=np.nonzero(own[1:]!=own[:-1])
        for d in (-1,0,1):
            y=2*yy+2;x=2*xx+1+d;seam[y,x]=True;ca[y,x]=own[yy,xx];cb[y,x]=own[yy+1,xx]
        ownbits=code_bits(torch.from_numpy(own.astype(np.int64))[None])[0]
        if not seam.any():
            result.append(F.avg_pool2d(ownbits*127,4));records.append(dict(junction_third_code_fraction=0.));continue
        distance,nearest=cv2.distanceTransformWithLabels((~seam).astype(np.uint8),cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
        assert np.array_equal(nearest[seam],np.arange(1,seam.sum()+1))
        idx=nearest[1::2,1::2]-1;aa=ca[seam][idx];bb=cb[seam][idx]
        other=np.where(aa==own,bb,aa)
        # At a triple junction the nearest segment may have neither own code.
        # Keep the own colour and select its nearest segment's first colour;
        # record this ambiguous subset explicitly rather than calling it exact.
        third=(aa!=own)&(bb!=own)
        obits=code_bits(torch.from_numpy(other.astype(np.int64))[None])[0]
        # Bound the common mixture BEFORE averaging. Extrapolation beyond
        # either endpoint can break cumulative-code ordering when neighbouring
        # native pixels select different interfaces inside one4x4 footprint.
        d=(torch.from_numpy(distance[1::2,1::2].copy())*.5).clamp_max(63.5/gain)
        field=63.5*(ownbits+obits)+gain*d[None]*(ownbits-obits)
        result.append(F.avg_pool2d(field,4).clamp(0,127))
        records.append(dict(junction_third_code_fraction=float(third.mean())))
    out=torch.stack(result).to(code.device)
    return (out,records) if with_stats else out


def regression():
    for value in (0,481,1214):
        code=torch.full((1,16,24),value,dtype=torch.long);out=joint_interface_fields(code)
        assert torch.equal(out,code_bits(code)[:,:,:4,:6]*127)
    code=torch.zeros(1,32,64,dtype=torch.long);code[...,32:]=1214
    out=joint_interface_fields(code);expected=(63.5+8*(torch.arange(64)-31.5))[None,None,None].expand(1,30,32,64)
    expected=F.avg_pool2d(expected.clamp(0,127),4)
    assert torch.equal(out,expected)
    assert torch.equal(joint_interface_fields(1214-code),127-out)
    for lo,hi in GROUPS30:assert (out[:,lo:hi-1]>=out[:,lo+1:hi]).all()
    generator=torch.Generator().manual_seed(129)
    code=torch.randint(0,1215,(2,12,16),generator=generator).repeat_interleave(4,-2).repeat_interleave(4,-1)
    out=joint_interface_fields(code)
    for lo,hi in GROUPS30:assert (out[:,lo:hi-1]>=out[:,lo+1:hi]-1e-5).all()
    print('JOINT_INTERFACE_PASS constants/half_pixel_boundary/common_fields/complement/order',flush=True)


if __name__=='__main__':torch.set_num_threads(2);regression()
