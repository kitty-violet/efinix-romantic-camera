"""Keep real header fetch within its existing bias/multiplier/ReLU windows."""
import json,struct
from .validate_region_native import coefficient_image as direct_image,align
from .runtime import file_hash,write_json


def coefficient_image(package,root):
    ops=direct_image(package,root);path=root/'coefficients-padded.bin';memory=bytearray(path.read_bytes())
    bias,mult,relu=12608,15040,17472
    for n in package['plan']['nodes']:
        if not n['kernel']:continue
        name=n['id'];p=package['parameters'][name];co=n['channels'];pad=(co+7)//8*8;op=ops[name]
        bias=align(bias);mult=align(mult)
        assert bias+pad*4<=15040 and mult+pad*4<=17472 and relu+pad//8<=17600
        op.update(bias_offset=bias,multiplier_offset=mult,relu_offset=relu)
        for offset,key in [(bias,'bias_int32'),(mult,'multiplier_q17')]:
            for c,v in enumerate(p[key]+[0]*(pad-co)):
                struct.pack_into('<i',memory,offset+4*c,int(v));assert struct.unpack_from('<i',memory,offset+4*c)[0]==v
        for g in range(pad//8):memory[relu+g]=255 if n['activation']=='relu' else 0
        bias+=pad*4;mult+=pad*4;relu+=pad//8
    path.write_bytes(memory)
    (root/'load_words.hex').write_text(''.join(f'{int.from_bytes(memory[i:i+8],"little"):016x}\n' for i in range(0,len(memory),8)),encoding='ascii')
    write_json(root/'coefficient_map.json',dict(schema='region-fixed-header-windows-v2',operators=list(ops.values()),bytes=len(memory),used_bytes=relu,
        sha256=file_hash(path),production_B1_image=False,existing_header_windows=True,bias_end=bias,multiplier_end=mult,relu_end=relu))
    return ops
