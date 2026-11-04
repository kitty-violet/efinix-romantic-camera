"""B1 specialization restrictions shared by descriptor generation and tests.

Geometry only; weights, bias, quantization scales and main_scaled stay runtime.
The public field layout is unchanged. RTL must reject these illegal geometries.
"""
import argparse,json
from pathlib import Path
from resource_route_contract import check_fields as generic_fields
from run_units import ROOT,sha

VERSION='B1_SPECIALIZED_V1'

def cyclic(value,rows):
    first=value&3
    return first<3 and all((value>>(2*y))&3==(first+y)%3 for y in range(rows))

def check_fields(f):
    mapping=generic_fields(f)
    if f[0]!=2:
        assert f[3]==(1 if f[6] else 4),'DMA supports only RGB/C32'
        return None
    assert cyclic(f[13],f[3]),'cyclic output planes'
    rgb=bool(f[6]&1) and f[28]==7 and f[4]==f[5]==1
    assert f[28]==255 or rgb,'fixed lane masks'
    if f[6]&16:
        assert f[4]==f[5]==4,'C32 fused residual'
        assert f[23] in (0,2),'residual offset 0/2'
        assert cyclic(f[20],f[3]),'cyclic skip planes'
    n=len(mapping);p=[(f[10]>>(2*y))&3 for y in range(n)]
    packed=f[11]|(f[12]<<32);b=[(packed>>(10*y))&1023 for y in range(n)]
    for y in range(1,n):
        if p[y]==p[y-1]:assert b[y]==b[y-1],'adjacent repeated row base'
        else:assert p[y]==(p[y-1]+1)%3,'cyclic input transition'
    # Nonadjacent repeated plane belongs to its other physical row slot.
    ordinal=0;seen={}
    for y in range(n):
        if y and p[y]!=p[y-1]:ordinal+=1
        slot=p[y]*2+(ordinal>=3)
        assert slot==mapping[y],'canonical row slot'
        if slot not in seen:seen[slot]=b[y]
    return mapping

def check_program(path):
    counts={'compute':0,'dma':0};maps=set()
    for i,line in enumerate(path.read_text().splitlines()):
        word=int(line,16);f=[(word>>(32*j))&0xffffffff for j in range(32)]
        try:m=check_fields(f)
        except AssertionError as exc:raise AssertionError(f'program[{i}]: {exc}') from exc
        counts['compute' if m is not None else 'dma']+=1
        if m is not None:maps.add(m)
    return dict(version=VERSION,commands=counts,row_maps=len(maps))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    program=ROOT/'tmp/frame-execution/joint-reorder-vectors1/program.hex'
    result=check_program(program);rows=[]
    for line in program.read_text().splitlines():
        w=int(line,16);rows.append([(w>>(32*j))&0xffffffff for j in range(32)])
    bad=[]
    def mutation(f,index,value):
        f=f.copy();f[index]=value;bad.append(f)
    feature=next(f for f in rows if f[0]!=2 and f[6]==0)
    rgb=next(f for f in rows if f[0]!=2 and f[6]==1)
    for groups in (1,2):mutation(feature,3,groups)
    for groups in (2,4):mutation(rgb,3,groups)
    fused=next(f for f in rows if f[0]==2 and f[6]&16 and f[3]==4)
    mutation(fused,23,1);mutation(fused,23,3);mutation(fused,4,2);mutation(fused,5,2)
    mutation(fused,13,0);mutation(fused,20,0);mutation(fused,28,1)
    source=next(f for f in rows if f[0]==2 and len(generic_fields(f))==6 and f[10]==0x924)
    mutation(source,10,0x909)
    for i,f in enumerate(bad):
        try:check_fields(f)
        except AssertionError:continue
        raise AssertionError(f'mutation {i} was accepted')
    result.update(status='SPECIALIZED_GEOMETRY_CONTRACT_PASS',rejected_mutations=len(bad),
        program_sha256=sha(program),network_contract_sha256=sha(ROOT/'network_ab/model_b_contract.json'),
        script_sha256=sha(__file__),runtime_numeric_parameters_unchanged=True)
    assert not a.output.exists();a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()
