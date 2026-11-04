"""Exercise real learned coefficients on the existing immutable RTL binary.

No hardware sources are modified. Inputs are distinct synthetic SRAM contents,
not video. This is an actual coefficient-load/numeric/protocol local test and
does not establish a complete model graph, style, fit, timing or board pass.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import time
import numpy as np
from .runtime import file_hash,write_json
from .region_detail_target import ROOT,RUN

PROJECT=ROOT.parent
NATIVE=PROJECT/'tmp/frame-execution/dsp-quant-add-cluster2'
BIN=PROJECT/'tools/verilator/ucrt64/bin'


def scalar_rq(v,m,relu=False):
    assert -(1<<31)<=v<(1<<31)
    z=int(v)*int(m);y=(abs(z)+65536)//131072
    if z<0:y=-y
    return max(0 if relu else -128,min(127,y))

def byte(bank,address,lane,case):return ((bank*37+address*13+lane*29+case*11)&255)-128
def pack(values,bits):return sum(int(v)<<(i*bits) for i,v in enumerate(values))
def align(x):return (x+63)//64*64


def coefficient_image(package,root):
    # Existing phase reader accepts weight offsets only in [1792,12608).
    memory=bytearray(20480);memory[:8]=b'REGION01';cursor=1792;ops={};records=[]
    for n in package['plan']['nodes']:
        if not n['kernel']:continue
        p=package['parameters'][n['id']];co=n['channels'];ci=n['input_channels'];ocpad=(co+7)//8*8
        w=np.asarray(p['weight_oihw'],np.int64);cursor=align(cursor);start=cursor;flat=[]
        for b in range(0,ocpad,8):
            if n['kind']=='dw3x3':
                positions=[(0,y,x) for y in range(3) for x in range(3)]
            else:
                positions=[(c,y,x) for cb in range(0,ci,8) for y in range(n['kernel']) for x in range(n['kernel']) for c in range(cb,min(ci,cb+8))]
            for c,y,x in positions:
                for lane in range(8):
                    oc=b+lane;v=int(w[oc,c,y,x]) if oc<co else 0;memory[cursor]=v&255;cursor+=1;flat.append((oc,c,y,x,v))
        assert cursor<=12608
        op=dict(name=n['id'],weight_offset=start,weight_bytes=cursor-start)
        for field,key in [('bias','bias_int32'),('multiplier','multiplier_q17')]:
            cursor=align(cursor);op[field+'_offset']=cursor
            vals=p[key]+[0]*(ocpad-co)
            for v in vals:struct.pack_into('<i',memory,cursor,int(v));cursor+=4
        cursor=align(cursor);op['relu_offset']=cursor
        for b in range(0,ocpad,8):memory[cursor]=((1<<min(8,co-b))-1) if n['activation']=='relu' else 0;cursor+=1
        # Independent readback of the actual serialized tensor, including padding.
        for i,(oc,c,y,x,v) in enumerate(flat):
            got=struct.unpack_from('<b',memory,start+i)[0];assert got==v
        ops[n['id']]=op;records.append(op)
    used=align(cursor);assert used<=20480
    raw=bytes(memory);(root/'coefficients-padded.bin').write_bytes(raw)
    (root/'load_words.hex').write_text(''.join(f'{int.from_bytes(raw[i:i+8],"little"):016x}\n' for i in range(0,len(raw),8)),encoding='ascii')
    write_json(root/'coefficient_map.json',dict(schema='research-direct-offset-coefficient-image-v1',operators=records,used_bytes=used,
        bytes=20480,sha256=hashlib.sha256(raw).hexdigest(),production_B1_image=False))
    return ops


def vectors(package,root):
    root.mkdir(exist_ok=False);ops=coefficient_image(package,root);nodes=package['plan']['nodes'];byid={n['id']:n for n in nodes}
    additions={n['inputs'][0]:n for n in nodes if n['kind']=='residual_add'}
    shapes=[]
    for n in nodes:
        if not n['kernel']:continue
        shapes.append((n['id'],32 if n['stride']==2 else 64,2 if n['stride']==2 else 4,16,4))
    # Frozen TB case9 exercises cancel and invalid-direction residual returns.
    shapes.insert(9,(next(iter(additions)),80,2,160,3))
    # A final packed RGB case also runs interruption/recovery tests.
    shapes.append((nodes[-1]['id'],7,3,17,6))
    gold=[];cases=[];details=[];fields={n:[] for n in ('fuse','main_scaled','skip_planes','skip_bases','skip_delta','align','post','stem','stride','view','bias','multiplier','relu')}
    for cid,(name,w,h,ox,oy) in enumerate(shapes):
        n=byid[name];p=package['parameters'][name];weights=np.asarray(p['weight_oihw'],np.int64);op=ops[name]
        pw=n['kind']=='pw1x1';stem=n['kind']=='conv3x3';stride=n['stride']==2;nn=byid.get(n['inputs'][0],{}).get('kind')=='nearest_x2'
        ci=n['input_channels'];co=n['channels'];gi=(ci+7)//8;go=(co+7)//8;sx=ox*(2 if stride else 1);sy0=oy*(2 if stride else 1)
        sh,sw,_=n['input_shape_hwc'];origin=(sx if pw else max(0,sx-1))>>int(nn)
        span=((w+3 if pw else (2*w-1 if stride else w)+5)//4)*gi;outspan=(w+3)//4*go
        if nn:span=((min(sw-1,ox+w)//2-origin+4)//4)*gi
        count=[0]*3;allocated={};planes=[];bases=[];opl=[];oba=[];outcount=[0]*3
        for y in range(6):
            sy=(sy0+y if pw else max(0,min(sh-1,sy0+y-1)))>>int(nn);plane=(sy+cid)%3
            if sy not in allocated:allocated[sy]=count[plane]*span;count[plane]+=1
            planes.append(plane);bases.append(allocated[sy])
        for y in range(4):
            plane=(oy+y+cid+1)%3;opl.append(plane);oba.append(2*span+16+outcount[plane]*outspan);outcount[plane]+=1
        if cid==9:
            planes=[0,1,0,1,0,1];bases=[(y//2)*span for y in range(6)];opl=[2,0,1,2];oba=[16+y*outspan for y in range(4)]
            # Plane0 output row1 must not alias the live input rows0/2.
            oba[1]=3*span+16
        assert max(oba)+outspan<620
        skip_planes=[(y+cid+2)%3 for y in range(4)];skip_bases=[620+(y//3)*128 for y in range(4)];delta=(cid%2)*2
        fused=name in additions;res=package['residual'][additions[name]['id']] if fused else dict(main_scaled=False,multiplier=0,post_multiplier=0)
        values=dict(fuse=int(fused),main_scaled=int(res['main_scaled']),skip_planes=pack(skip_planes,2),skip_bases=pack(skip_bases,10),skip_delta=delta,
            align=res['multiplier'],post=res['post_multiplier'],stem=int(stem),stride=int(stride),view=int(nn),
            bias=pack([v&0xffffffff for v in p['bias_int32']],32),multiplier=pack(p['multiplier_q17'],32),relu=(1<<co)-1 if n['activation']=='relu' else 0)
        for key,v in values.items():fields[key].append(v)
        offset=len(gold);cases.append([cid,offset,int(pw),w,h,gi,go,(1<<min(8,co))-1,sx,origin,sw,span,op['weight_offset'],pack(planes,2),pack(bases,10),pack(opl,2),pack(oba,10)])
        for y in range(h):
            for x in range(w):
                for group in range(go):
                    vals=[]
                    for lane in range(8):
                        oc=group*8+lane
                        if oc>=co:vals.append(0);continue
                        dot=0
                        if pw:
                            for ch in range(ci):dot+=byte(planes[y]*4+x%4,bases[y]+x//4*gi+ch//8,ch%8,cid)*int(weights[oc,ch,0,0])
                        else:
                            for ky in range(3):
                                for kx in range(3):
                                    ry=(2*y if stride else y)+ky;lx=(max(0,min(sw-1,sx+(2*x if stride else x)+kx-1))>>int(nn))-origin
                                    bank=planes[ry]*4+lx%4;address=bases[ry]+lx//4*gi+(0 if stem else group)
                                    if stem:
                                        for ch in range(3):dot+=byte(bank,address,ch,cid)*int(weights[oc,ch,ky,kx])
                                    else:dot+=byte(bank,address,lane,cid)*int(weights[oc,0,ky,kx])
                        v=scalar_rq(dot+p['bias_int32'][oc],p['multiplier_q17'][oc],n['activation']=='relu')
                        if fused:
                            s=byte(skip_planes[y]*4+(x+delta)%4,skip_bases[y]+(x+delta)//4*go+group,lane,cid)
                            a,b=(v,s) if res['main_scaled'] else (s,v)
                            v=scalar_rq(scalar_rq(a,res['multiplier'])+b,res['post_multiplier'])
                        vals.append(v&255)
                    gold.append(pack(vals,8))
        details.append(dict(case=cid,node=name,width=w,rows=h,fused=fused,nn=nn,stride2=stride))
    (root/'golden.hex').write_text(''.join(f'{v:016x}\n' for v in gold))
    (root/'cases.txt').write_text(str(len(cases))+'\n'+''.join(' '.join(str(v) if i<13 else f'{v:x}' for i,v in enumerate(r))+'\n' for r in cases))
    for k,values in fields.items():
        width=256 if k in ('bias','multiplier') else 16
        (root/f'case_{k}.hex').write_text(''.join(f'{v:0{width}x}\n' for v in values))
    result=dict(cases=len(cases),words=len(gold),scalar_checks_per_mode=len(gold)*8,details=details,learned_weights=True,synthetic_activations=True)
    write_json(root/'oracle.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path);p.add_argument('--name');p.add_argument('--check-readback',type=Path);p.add_argument('--vectors',type=Path);p.add_argument('--rtl-build',type=Path);a=p.parse_args()
    if a.check_readback:
        actual=b''.join(int(s,16).to_bytes(8,'little') for s in a.check_readback.read_text().split());expected=(a.vectors/'coefficients-padded.bin').read_bytes()
        assert actual==expected;print('ACTUAL_RTL_READBACK_SOFTWARE_VERIFIED',flush=True);return
    assert a.package and a.name;root=ROOT/'runs'/RUN/a.name;assert root.parent==ROOT/'runs'/RUN;root.mkdir(exist_ok=False)
    package=json.loads(a.package.read_text());oracle=vectors(package,root/'vectors');native=json.loads(((a.rtl_build/'binary.json') if a.rtl_build else NATIVE/'native_inputs.json').read_text())
    binary=Path(native['binary']);assert file_hash(binary)==native['binary_sha256']
    frozen=(a.rtl_build or NATIVE)/'frozen'
    if not a.rtl_build:assert file_hash(frozen/'inputs.json')==native['source_manifest_sha256']
    bound={str(f):file_hash(f) for f in frozen.iterdir() if f.is_file()}
    write_json(root/'inputs.json',dict(binary=str(binary),binary_sha256=file_hash(binary),hardware_sources=bound,package=str(a.package),
        package_sha256=file_hash(a.package),generator_sha256=file_hash(__file__),oracle=oracle,max_host_seconds=60,full_frame=False))
    dump=root/'readback.hex';verify=subprocess.list2cmdline([sys.executable,'-B','-m','network_ab.training.validate_region_native','--check-readback',str(dump),'--vectors',str(root/'vectors')])
    env=os.environ.copy();env['PATH']=str(BIN)+';'+env['PATH'];env['PYTHONPATH']=str(PROJECT);env['TEMP']=str(root);env['TMP']=str(root)
    start=time.perf_counter()
    with (root/'simulation.log').open('w',encoding='utf-8') as f:
        status=subprocess.run([str(binary),'+VECTORS='+(root/'vectors').as_posix(),'+DUMP='+dump.as_posix(),'+VERIFY_COMMAND='+verify],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=60)
    body=(root/'simulation.log').read_text(encoding='utf-8');passed='PASS COMPUTE_CLUSTER' in body and 'ACTUAL_RTL_READBACK_SOFTWARE_VERIFIED' in body and status.returncode==0 and 'FAIL ' not in body
    write_json(root/'result.json',dict(status='TRAINED_WEIGHTS_LOCAL_RTL_PASS' if passed else 'FAIL',seconds=time.perf_counter()-start,exit_code=status.returncode,
        oracle=oracle,lines=[s for s in body.splitlines() if s.startswith(('BLOCK ','PASS ','FAIL '))],log_sha256=file_hash(root/'simulation.log'),
        complete_graph=False,fit=False,timing=False,board_release=False))
    assert all(file_hash(p)==h for p,h in bound.items());assert passed,body[-5000:]
    print('TRAINED_LOCAL_RTL_PASS',oracle['cases'],flush=True)

if __name__=='__main__':main()
