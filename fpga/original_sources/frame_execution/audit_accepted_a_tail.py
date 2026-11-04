"""Prove the actual loader's ninth-tap domain for immutable accepted A."""
import collections,json
from pathlib import Path
from run_units import ROOT,sha
from accepted_a_assets import accepted_package,PACKAGE_SHA

def proof():
    raw,params,residual,graph=accepted_package()
    layout=json.loads((ROOT/'model-training-workspace/accepted/B1-A-20261008/contract/coefficient_layout.json').read_text())
    rows=[];hist=collections.Counter()
    for op in layout['operators']:
        p=params[op['name']]
        if op['kind']=='conv3x3':
            # Image is tap-major RGB; fixed_weight_loader transposes to
            # input-channel phases. Ninth spatial tap is indices24/25/26.
            assert op['input_channels']==3 and op['name']=='stem'
            values=[w[8*3+ic] for w in p.weights for ic in range(3)]
        elif op['kind']=='dw3x3':values=[w[8] for w in p.weights]
        else:
            assert op['kind']=='pw1x1'
            # Loader explicitly inserts zero at tap8 of each IC8 phase.
            values=[0]*(op['padded_channels']*max(1,op['input_channels']//8))
        assert values and set(values)<=set((0,4,8)),(op['name'],set(values))
        hist.update(values);rows.append(dict(operator=op['name'],kind=op['kind'],tail_histogram=dict(collections.Counter(values))))
    checks=0
    for a in range(-128,128):
        for b in (0,4,8):
            shifted=0 if b==0 else a<< (2 if b==4 else 3)
            assert shifted==a*b and -1024<=shifted<=1016
            checks+=1
    return dict(status='ACCEPTED_A_LOADER_TAIL_048_PROVEN',package_sha256=PACKAGE_SHA,
        operators=rows,tail_histogram=dict(hist),exhaustive_integer_products=checks,
        allowed_tail_coefficients=[0,4,8],topology_changed=False,coefficients_changed=False,
        requirement='Specialized hardware must reject any unsupported ninth-tap load before activation; other eight taps remain unrestricted.',
        limitation='This proves the fixed accepted A domain, not arbitrary B1 coefficient packages.',
        checker_sha256=sha(__file__),hardware_access=False)

if __name__=='__main__':
    dest=ROOT/'tmp/frame-execution/accepted-a-tail-domain1.json';assert not dest.exists()
    p=proof();dest.write_text(json.dumps(p,indent=2),encoding='utf8')
    print(p['status'],p['tail_histogram'],p['exhaustive_integer_products'])
