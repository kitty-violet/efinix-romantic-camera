"""CPU-only checks of the exact integer mask coefficients in a saved package."""
import argparse
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from .b1_trained_image import decode


def quantize(dot,m,positive=True):
    z=dot.astype(np.int64)*int(m)
    y=np.sign(z)*((np.abs(z)+65536)//131072)
    return y.clip(0 if positive else -128,127).astype(np.int32)


def dw(a,p,c):
    windows=np.lib.stride_tricks.sliding_window_view(np.pad(a,1,mode='edge'),(3,3))
    w=np.array(p.weights[c],np.int64).reshape(3,3)
    return quantize((windows*w).sum((-1,-2))+p.bias[c],p.multiplier[c])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();raw=Path(a.package).read_bytes();params,residual=decode(raw)
    assert residual['res3_add']==dict(main_scaled=False,multiplier=1,post_multiplier=131071)
    levels=[(0,q) for q in (14,28,42,56,70,84,98,112)]
    levels += [(4,q) for q in (49,59,69,79)]+[(8,q) for q in (49,59,69,79)]
    mask=np.zeros((48,64),np.int32);mask[8:22,7:25]=1;mask[32,10]=1
    mask[28:44,34:35]=1;mask[29:40,44:58]=1;records=[]
    threshold=params['res3_conv1.pw'];identity=params['res3_conv2.pw']
    for c,(j,q) in enumerate(levels):
        weights=np.array(threshold.weights[c],np.int64)
        assert np.count_nonzero(weights)==1 and weights[j]!=0
        ramp=quantize(np.arange(128)*weights[j]+threshold.bias[c],threshold.multiplier[c])
        assert np.array_equal(ramp,(np.arange(128)>=q).astype(np.int32)*127),(c,ramp)
        bits=mask*127
        mid=dw(bits,params['res3_conv2.dw'],c)
        iw=np.array(identity.weights[c],np.int64);assert np.count_nonzero(iw)==1 and iw[c]!=0
        mid=quantize(mid*iw[c]+identity.bias[c],identity.multiplier[c],False)
        mid=quantize(mid,residual['res3_add']['post_multiplier'],False)
        out=dw(mid,params['up1_low_dw'],c)
        closing=c>=8 and q<64
        op=cv2.MORPH_CLOSE if closing else cv2.MORPH_OPEN
        expected=cv2.morphologyEx(bits.astype(np.uint8),op,np.ones((3,3),np.uint8),borderType=cv2.BORDER_REPLICATE)
        assert np.array_equal(out,expected),(c,np.max(np.abs(out-expected)))
        cc,lab=cv2.connectedComponents((out>0).astype(np.uint8),connectivity=4)
        areas=np.bincount(lab.ravel())[1:]
        if not closing:assert areas.min()>=9 and out[32,10]==0 and out[34,34]==0
        records.append(dict(channel=c,closing=closing,mask_states=np.unique(out).tolist(),component_areas=areas.tolist()))
    report=dict(status='PASS',package_sha256=hashlib.sha256(raw).hexdigest(),records=records,
        scope='Exact16 integer mask thresholds and opening/closing; not full-network or RGB minimum-area acceptance',
        thin_structures_removed_by_opening=True,visual_acceptance=False)
    Path(a.output).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('CPU_INTEGER_MASK_CHECK_PASS',a.output,flush=True)


if __name__=='__main__':main()
