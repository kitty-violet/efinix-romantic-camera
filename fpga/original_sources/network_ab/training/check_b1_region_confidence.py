"""Exact CPU constant-region propagation through the saved integer decoder."""
import argparse
from pathlib import Path
import numpy as np
from .b1_trained_image import decode
from .check_b1_mask_opening import quantize
from .runtime import write_json,file_hash


def run(params,residual,plane,value):
    z=np.zeros(32,np.int64);z[plane]=value;records={}
    for key,dw in [('res3_conv2.dw',True),('res3_conv2.pw',False),('up1_low_dw',True),
                   ('up1_low_pw',False),('up1_high_dw',True),('up2_low_dw',True),('up2_low_pw',False),('up2_high_dw',True)]:
        p=params[key];w=np.array(p.weights,np.int64).reshape(len(p.bias),-1)
        dot=(w.sum(1)*z if dw else w@z)+np.array(p.bias,np.int64)
        z=np.array([quantize(np.array(v),p.multiplier[c]).item() for c,v in enumerate(dot)],np.int64)
        if key=='res3_conv2.pw':z=quantize(z,residual['res3_add']['post_multiplier']).astype(np.int64)
        records[key]=z.tolist()
    return records


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    params,residual=decode(Path(a.package).read_bytes());records=[]
    for plane,positive in ((0,True),(8,False)):
        for margin in (1.5,4.):
            sign=1 if positive else -1;raw=63.5+sign*24*margin
            value=int(np.clip(np.floor(raw+.5),0,127));got=run(params,residual,plane,value)
            wanted=run(params,residual,plane,127 if positive else 0)
            error=max(abs(x-y) for x,y in zip(got['up2_high_dw'],wanted['up2_high_dw']))
            records.append(dict(plane=plane,true_region=positive,logit_margin=margin,classifier_coverage=value,
                final_features=got['up2_high_dw'],desired_features=wanted['up2_high_dw'],feature_error=error))
    assert all(r['feature_error']==0 for r in records if r['logit_margin']==4.)
    assert any(r['feature_error']>0 for r in records if r['logit_margin']==1.5)
    write_json(a.output,dict(records=records,package_sha256=file_hash(a.package),
        scope='Constant intermediate-mask diagnostic; not source inference or a visual acceptance test'))
    print('REGION_CONFIDENCE_CHECK',records,flush=True)


if __name__=='__main__':main()
