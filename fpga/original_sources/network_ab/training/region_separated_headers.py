"""Pack directly into existing separate coefficient windows, without gaps
from a discarded interleaved image. This changes offsets, not header format.
"""
import struct
import numpy as np
from .runtime import file_hash,write_json


def coefficient_image(package,root):
    memory=bytearray(20480);memory[:8]=b'REGION01';used=set();ops={};cursor=1792;bias=12608;mult=15040;relu=17472
    def put(offset,payload,lo,hi):
        assert lo<=offset and offset+len(payload)<=hi
        addresses=set(range(offset,offset+len(payload)));assert not addresses&used
        memory[offset:offset+len(payload)]=payload;used.update(addresses)
    for n in package['plan']['nodes']:
        if not n['kernel']:continue
        p=package['parameters'][n['id']];ci=n['input_channels'];co=n['channels'];pad=(co+7)//8*8;w=np.asarray(p['weight_oihw'],np.int64)
        assert np.all((-127<=w)&(w<=127))
        cursor=(cursor+63)//64*64;start=cursor;serialized=[]
        for oc8 in range(0,pad,8):
            if n['kind']=='dw3x3':positions=[(0,y,x) for y in range(3) for x in range(3)]
            else:positions=[(c,y,x) for cb in range(0,ci,8) for y in range(n['kernel']) for x in range(n['kernel']) for c in range(cb,min(ci,cb+8))]
            for c,y,x in positions:
                for lane in range(8):serialized.append(int(w[oc8+lane,c,y,x]) if oc8+lane<co else 0)
        put(cursor,bytes(v&255 for v in serialized),1792,12608);cursor+=len(serialized)
        bias=(bias+63)//64*64;mult=(mult+63)//64*64
        op=dict(name=n['id'],weight_offset=start,weight_bytes=len(serialized),bias_offset=bias,multiplier_offset=mult,relu_offset=relu)
        for offset,values,lo,hi in [(bias,p['bias_int32'],12608,15040),(mult,p['multiplier_q17'],15040,17472)]:
            vals=values+[0]*(pad-co);put(offset,struct.pack('<'+'i'*pad,*vals),lo,hi)
            assert list(struct.unpack_from('<'+'i'*pad,memory,offset))==vals
        put(relu,bytes([255 if n['activation']=='relu' else 0]*(pad//8)),17472,17600)
        # Read back into original OIHW coordinates independently of write loop.
        actual=np.zeros_like(w)
        for oc in range(co):
            for ic in range(w.shape[1]):
                for yy in range(n['kernel']):
                    for xx in range(n['kernel']):
                        term=yy*3+xx if n['kind']=='dw3x3' else (ic//8*n['kernel']**2+yy*n['kernel']+xx)*min(ci,8)+ic%8
                        address=start+((oc//8)*n['terms']+term)*8+oc%8
                        actual[oc,ic,yy,xx]=struct.unpack_from('<b',memory,address)[0]
        assert np.array_equal(actual,w),n['id']
        bias+=pad*4;mult+=pad*4;relu+=pad//8;ops[n['id']]=op
    path=root/'coefficients-padded.bin';path.write_bytes(memory)
    (root/'load_words.hex').write_text(''.join(f'{int.from_bytes(memory[i:i+8],"little"):016x}\n' for i in range(0,len(memory),8)),encoding='ascii')
    write_json(root/'coefficient_map.json',dict(schema='region-fixed-header-direct-separated-v3',operators=list(ops.values()),bytes=len(memory),used_bytes=relu,sha256=file_hash(path),production_B1_image=False,existing_header_windows=True,weight_end=cursor,bias_end=bias,multiplier_end=mult,relu_end=relu,overlap_checked=True,oihw_readback_checked=True))
    return ops
