"""Portable read-only S4R1 package verifier. Python standard library only."""
from pathlib import Path
import json,hashlib,struct,zlib,argparse,math

def read(p):return json.loads(p.read_text('utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def clamp(v,lo=-128,hi=127):return max(lo,min(hi,v))
def rq(v,m):
    n=v*m;value=(abs(n)+65536)//131072
    return -value if n<0 else value

def load(folder):
    graph=read(folder/'graph.json');layout=read(folder/'layout.json');raw=(folder/'coefficients-padded.bin').read_bytes()
    assert raw[:8]==b'S4R1PKG1' and struct.unpack_from('<6I',raw,8)==(1,1280,720,11,1,3)
    canon=json.dumps(graph,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')
    assert raw[32:64]==hashlib.sha256(canon).digest() and raw[32:64].hex()==layout['graph_sha256']
    assert len(raw)==8192 and not any(raw[layout['effective_bytes']:])
    assert (folder/'coefficients.bin').read_bytes()==raw[:layout['effective_bytes']]
    assert hashlib.sha256(raw).hexdigest()==layout['sha256']
    assert f'{zlib.crc32(raw):08x}'==layout['crc32']
    lines=(folder/'load_words.hex').read_text('ascii').splitlines()
    assert len(lines)==1024 and b''.join(int(x,16).to_bytes(8,'little') for x in lines)==raw
    parameters={};weight_count=bias_count=0
    for row in layout['operators']:
        ps={}
        for kind,f in row['fields'].items():
            fmt='b' if f['dtype']=='i1' else 'i';size=1 if fmt=='b' else 4;count=math.prod(f['shape'])
            assert f['length']==count*size and f['offset']%8==0
            ps[kind]=struct.unpack_from('<'+str(count)+fmt,raw,f['offset'])
        assert all(-127<=w<=127 for w in ps['weight'])
        assert all(0<m<=131071 for m in ps['multiplier'])
        parameters[row['node']]=ps;weight_count+=len(ps['weight']);bias_count+=len(ps['bias'])
    assert weight_count==3808 and bias_count==227 and len(parameters)==11 and len(graph['nodes'])==13
    for i,name in enumerate(['res1_add']):
        a,b,c=struct.unpack_from('<3i',raw,layout['residual_offset']+12*i)
        assert dict(main_scaled=bool(a),multiplier=b,post_multiplier=c)==graph['residual'][name]
    return graph,parameters

def oracle(graph,parameters,a):
    tensors={'input':[[[int(v)-128 for v in p] for p in row] for row in a]}
    for r in graph['nodes']:
        name=r['id'];x=tensors[r['inputs'][0]];h=len(x);w=len(x[0]);cin=len(x[0][0])
        if name in parameters:
            p=parameters[name];s=r['stride'];k=r['kernel'];cout=r['channels'];out=[];depthwise=r['groups']==cin and cin>1
            ci=1 if depthwise else cin
            for y in range((h+s-1)//s):
                row=[]
                for xx in range((w+s-1)//s):
                    pixel=[]
                    for oc in range(cout):
                        total=0
                        for ic0 in range(ci):
                            ic=oc if depthwise else ic0
                            for ky in range(k):
                                iy=clamp(y*s+ky-k//2,0,h-1)
                                for kx in range(k):
                                    ix=clamp(xx*s+kx-k//2,0,w-1)
                                    index=((oc*ci+ic0)*k+ky)*k+kx
                                    total+=x[iy][ix][ic]*p['weight'][index]
                        acc=clamp(total+p['bias'][oc],-(2**31),2**31-1);v=clamp(rq(acc,p['multiplier'][oc]))
                        pixel.append(max(0,v) if r['activation']=='relu' else v)
                    row.append(pixel)
                out.append(row)
        elif r['kind']=='nearest':
            f=r['factor'];out=[[list(x[y//f][xx//f]) for xx in range(w*f)] for y in range(h*f)]
        else:
            z=tensors[r['inputs'][1]];p=graph['residual'][name];out=[]
            for y in range(h):
                row=[]
                for xx in range(w):
                    pixel=[]
                    for c in range(cin):
                        small,other=(x[y][xx][c],z[y][xx][c]) if p['main_scaled'] else (z[y][xx][c],x[y][xx][c])
                        pixel.append(clamp(rq(clamp(rq(small,p['multiplier']))+other,p['post_multiplier'])))
                    row.append(pixel)
                out.append(row)
        tensors[name]=out
    return tensors['rgb']

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);p.add_argument('--full-reference',action='store_true');a=p.parse_args();root=a.root
    manifest=read(root/'manifest-sha256.json')
    for rel,h in manifest.items():assert sha(root/rel)==h,rel
    catalog=read(root/'catalog.json');expected=int(catalog['accepted_or_retained_style_count']);assert expected>0
    checks=[]
    for row in catalog['styles']:
        folder=root/row['directory'];graph,parameters=load(folder);assert sha(folder/'coefficients-padded.bin')==row['coefficient_sha256']
        scalar=0
        if a.full_reference:
            for case in read(folder/'reference-vectors.json')['vectors']:
                out=oracle(graph,parameters,case['input_u8']);assert out==case['output_i8'],row['id'];scalar+=len(out)*len(out[0])*3
        checks.append(dict(id=row['id'],binary_hex_crc_layout=True,reference_output_scalars=scalar,parameters=4035))
    assert len(checks)==expected
    print(json.dumps(dict(status='PASS',style_count=expected,full_reference=a.full_reference,checks=checks,hardware_test=False),ensure_ascii=False))
if __name__=='__main__':main()
