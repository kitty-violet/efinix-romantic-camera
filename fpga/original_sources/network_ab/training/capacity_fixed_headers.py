"""Pack the complete larger parameter image into the existing 20-KiB store.

Weights, bias, multipliers and ReLU bytes use disjoint fixed reader windows.
No interleaved temporary bias copies are left in the weight subsection.
"""
import struct
import numpy as np
from .b1_capacity_component_net import budget,plan
from .runtime import file_hash,write_json


def coefficient_image(package,root):
    assert package['plan']==plan();b=budget();memory=bytearray(20480);memory[:8]=b'REGION01';ops={};written=set()
    def put(at,data):
        assert at>=1792 and at+len(data)<=17792
        slots=set(range(at,at+len(data)));assert not slots&written
        memory[at:at+len(data)]=data;written.update(slots)
    for n,row in zip([n for n in package['plan']['nodes'] if n['kernel']],b['operators']):
        assert n['id']==row['id'];p=package['parameters'][n['id']];w=np.asarray(p['weight_oihw'],np.int64)
        ci=n['input_channels'];co=n['channels'];pad=row['padded_channels'];raw=[]
        assert w.shape==(co,1 if n['kind']=='dw3x3' else ci,n['kernel'],n['kernel']) and w.min()>=-128 and w.max()<=127
        for oc in range(0,pad,8):
            positions=[(0,y,x) for y in range(3) for x in range(3)] if n['kind']=='dw3x3' else [(c,y,x) for block in range(0,ci,8) for y in range(n['kernel']) for x in range(n['kernel']) for c in range(block,min(ci,block+8))]
            for c,y,x in positions:
                for lane in range(8):raw.append(int(w[oc+lane,c,y,x])&255 if oc+lane<co else 0)
        put(row['weight_offset'],bytes(raw));op={k:v for k,v in row.items() if k.endswith('_offset')};op.update(name=n['id'],weight_bytes=len(raw));ops[n['id']]=op
        for label,key in [('bias','bias_int32'),('multiplier','multiplier_q17')]:
            assert len(p[key])==co;vals=p[key]+[0]*(pad-co);blob=struct.pack('<'+'i'*pad,*vals);put(row[label+'_offset'],blob)
            assert list(struct.unpack_from('<'+'i'*pad,memory,row[label+'_offset']))==vals
        put(row['relu_offset'],bytes([255 if n['activation']=='relu' else 0]*(pad//8)))
        assert list(memory[row['weight_offset']:row['weight_offset']+len(raw)])==raw
    assert not any(memory[17792:]);path=root/'coefficients-padded.bin';path.write_bytes(memory)
    (root/'load_words.hex').write_text(''.join(f'{int.from_bytes(memory[i:i+8],"little"):016x}\n' for i in range(0,len(memory),8)),encoding='ascii')
    write_json(root/'coefficient_map.json',dict(schema='capacity-disjoint-fixed-header-windows-v1',operators=list(ops.values()),bytes=len(memory),valid_image_bytes=17792,sha256=file_hash(path),no_section_overlap=True,budget=b,production_B1_image=False))
    return ops
