"""Integer F(4,3) identity and optimistic device-work bound, never RTL PASS.

Formula: Lavin/Gray arXiv:1509.09308, equation15. We scale G by24,
then divide the exact integer inverse by576. No FP or quantization change.
"""
import argparse, json, sys, random
from pathlib import Path
from run_units import ROOT, sha
sys.path.insert(0,str(ROOT/'network_ab'))
from model_contract import compile_contract,load_contract

B=[[4,0,-5,0,1,0],[0,-4,-4,1,1,0],[0,4,-4,-1,1,0],
   [0,-2,-1,2,1,0],[0,2,-1,-2,1,0],[0,4,0,-5,0,1]]
G=[[6,0,0],[-4,-4,-4],[ -4,4,-4],[1,2,4],[1,-2,4],[0,0,24]]
A=[[1,1,1,1,1,0],[0,1,-1,2,-2,0],[0,1,1,4,4,0],[0,1,-1,8,-8,1]]
def transpose(a):return list(map(list,zip(*a)))
def mm(a,b):return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]
def transform(a,b):return mm(mm(a,b),transpose(a))
def exact(d,g):
    u,v=transform(G,g),transform(B,d)
    y=transform(A,[[u[i][j]*v[i][j] for j in range(6)] for i in range(6)])
    assert all(x%576==0 for row in y for x in row)
    return [[x//576 for x in row] for row in y]
def reference(d,g):return [[sum(d[i+r][j+c]*g[r][c] for r in range(3) for c in range(3)) for j in range(4)] for i in range(4)]
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output.resolve();assert out.is_relative_to(ROOT/'tmp') and not out.exists()
    basis=0
    for di in range(36):
        for gi in range(9):
            d=[[int(6*r+c==di) for c in range(6)] for r in range(6)]
            g=[[int(3*r+c==gi) for c in range(3)] for r in range(3)]
            assert exact(d,g)==reference(d,g);basis+=1
    rng=random.Random(20261004)
    for k in range(1024):
        d=[[rng.randrange(-128,128) for _ in range(6)] for _ in range(6)]
        g=[[rng.randrange(-128,128) for _ in range(3)] for _ in range(3)]
        if k<4:
            d=[[(-128,127)[k&1]]*6 for _ in range(6)]
            g=[[(-128,127)[k>>1]]*3 for _ in range(3)]
        assert exact(d,g)==reference(d,g)
    plan=compile_contract(load_contract());rows=[]
    for n in plan['nodes']:
        mac=n['macs'];eligible=n['kind'] in ('conv3x3','dw3x3') and n['stride']==1
        h,w,c=n['output_shape_hwc'];ci=n['input_channels'] if n['kind']=='conv3x3' else 1
        transformed=((h+3)//4)*((w+3)//4)*c*ci*36 if eligible else 0
        # One 19x18 DSP product versus TWO ordinary8x8 MACs per physicalDSP.
        rows.append(dict(id=n['id'],eligible=eligible,direct_macs=mac,
                         direct_physical_dsp_cycles=mac/2,
                         candidate_physical_dsp_cycles=transformed if eligible else mac/2))
    direct=sum(x['direct_physical_dsp_cycles'] for x in rows)
    candidate=sum(x['candidate_physical_dsp_cycles'] for x in rows)
    # Conservative absolute interval bounds, including signed INT8=-128.
    vmax=128*max(sum(abs(x) for x in row) for row in B)**2
    umax=128*max(sum(abs(x) for x in row) for row in G)**2
    inverse=vmax*umax*max(sum(abs(x) for x in row) for row in A)**2*3
    result=dict(status='INTEGER_IDENTITY_AND_OPTIMISTIC_BOUND_ONLY',basis_pairs=basis,
                random_extreme_cases=1024,integer_convolution_identity_pass=True,
                full_quantized_network_executed=False,
                bounds=dict(transformed_activation_abs=vmax,transformed_weight_abs=umax,
                            stem_inverse_abs=inverse,stem_inverse_signed_bits=inverse.bit_length()+1),
                direct_physical_dsp_cycles=direct,candidate_physical_dsp_cycles=candidate,
                ideal_reduction_percent=(1-candidate/direct)*100,
                ideal_ms_by_compute_dsps={str(d):candidate/d/150000 for d in (64,96,128)},
                excluded=['transforms/inverse/divide576 hardware','wider weight and tile storage',
                          'input/output and internal RAM throughput','RQ/headers/DDR/turnaround',
                          'whole-board resource packing and timing'],nodes=rows,
                formula_source='https://arxiv.org/pdf/1509.09308',
                contract_sha256=sha(ROOT/'network_ab/model_b_contract.json'),
                script_sha256=sha(__file__),hardware_implemented=False,performance_pass=False)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k!='nodes'}))
if __name__=='__main__':main()
