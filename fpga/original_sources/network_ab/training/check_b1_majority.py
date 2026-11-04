"""Exhaustive local binary truth-table check of saved majority coefficients."""
import argparse
from pathlib import Path
import numpy as np
from .b1_trained_image import decode
from .check_b1_mask_opening import quantize
from .runtime import write_json,file_hash


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    params,residual=decode(Path(a.package).read_bytes())
    bits=((np.arange(512)[:,None]>>np.arange(9))&1).astype(np.int64)
    expected=(bits.sum(1)>=5).astype(np.int64)*127;count=0
    for key in ('res3_conv2.dw','up1_low_dw'):
        p=params[key]
        for c in range(16):
            weights=np.array(p.weights[c],np.int64).flatten()
            actual=quantize((bits*127)@weights+p.bias[c],p.multiplier[c])
            assert np.array_equal(actual,expected),(key,c)
            values=np.arange(128,dtype=np.int64)
            constant=quantize(values*weights.sum()+p.bias[c],p.multiplier[c])
            assert np.array_equal(constant,(9*values-508).clip(0,127))
            count+=512+128
    assert residual['res3_add']==dict(main_scaled=False,multiplier=1,post_multiplier=131071)
    write_json(a.output,dict(status='PASS',comparisons=count,package_sha256=file_hash(a.package),
        binary_rule='5or more of9 neighbours select the region',constant_endpoints_preserved=True,
        operations_added=0,scope='Local layer truth tables only; no global RGB minimum-area guarantee',visual_acceptance=False))
    print('MAJORITY_RULE_PASS',count,flush=True)


if __name__=='__main__':main()
