"""Bind the current production descriptor ABI to a real B1 style package.

Only three residual scale fields change. Program, addresses, dependencies and
DMA stream are preserved. No shared program or hardware source is overwritten.
"""
import argparse
import json
from pathlib import Path
import shutil
import zlib
from .b1_trained_image import decode
from .runtime import contract_guard,file_hash,write_json


def main(package):
    template=Path(r'E:\fpga\tmp\frame-execution\paired-ddr-descriptors1')
    frame=Path(r'E:\fpga\tmp\frame-execution\dma-credit-frame1\result.json')
    meta=json.loads((template/'result.json').read_text());reference=json.loads(frame.read_text())
    assert reference['descriptor_manifest_sha256']==file_hash(template/'result.json')
    assert file_hash(template/'schema.json')==meta['schema_sha256']
    assert all(file_hash(template/name)==sha for name,sha in meta['files'].items())
    assert meta['network_contract_sha256']==file_hash(Path(__file__).parents[1]/'model_b_contract.json')
    _,plan=contract_guard();_,residual=decode((package/'coefficients-padded.bin').read_bytes())
    nodes={n['index']:n for n in plan['nodes']};schema=json.loads((template/'schema.json').read_text())
    fields={f['field']:f for f in schema['compute']['fields']}
    def get(v,name):
        f=fields[name];return (v>>f['offset'])&((1<<f['width'])-1)
    def put(v,name,value):
        f=fields[name];assert 0<=value<1<<f['width'];mask=((1<<f['width'])-1)<<f['offset']
        return (v&~mask)|(value<<f['offset'])
    data=(template/'compute.bin').read_bytes();out=bytearray();counts={k:0 for k in residual}
    for offset in range(0,len(data),64):
        before=int.from_bytes(data[offset:offset+64],'little');after=before
        if get(before,'f6')&16:
            node=nodes[get(before,'f1')]['id'];r=residual[node]
            after=put(after,'f6',(get(before,'f6')&~32)|(int(r['main_scaled'])<<5))
            after=put(after,'f24',r['multiplier']);after=put(after,'f25',r['post_multiplier']);counts[node]+=1
            assert get(after,'f24')==r['multiplier'] and get(after,'f25')==r['post_multiplier']
        for field in fields:
            if field not in ('f6','f24','f25'):assert get(before,field)==get(after,field)
        assert ((get(before,'f6')^get(after,'f6'))&~32)==0
        out.extend(after.to_bytes(64,'little'))
    assert all(counts.values()) and len(out)==len(data)
    dest=package/'execution';dest.mkdir(exist_ok=False)
    (dest/'compute.bin').write_bytes(out);shutil.copy2(template/'dma.bin',dest/'dma.bin');shutil.copy2(template/'schema.json',dest/'schema.json')
    manifest=dict(status='TRAINED_SCALES_BOUND_TO_EXISTING_PRODUCTION_ABI',template=str(template),template_manifest_sha256=file_hash(template/'result.json'),
        template_full_frame_evidence=str(frame),template_full_frame_evidence_sha256=file_hash(frame),
        coefficient_sha256=file_hash(package/'coefficients-padded.bin'),residual=counts,scales=residual,
        compute_records=len(out)//64,dma_records=meta['dma_records'],changes=['f6 bit5 branch selection','f24 alignment Q17','f25 output Q17'],
        all_other_fields_unchanged=True,dma_byte_exact=True,actual_trained_full_frame_rtl=False,
        note='Execution buffers are derived from the fixed common program plus the three records inside the 20KiB style image; not extra learned weights or a new network graph.',
        files={name:dict(bytes=(dest/name).stat().st_size,sha256=file_hash(dest/name),crc32=zlib.crc32((dest/name).read_bytes())) for name in ('compute.bin','dma.bin','schema.json')})
    write_json(dest/'manifest.json',manifest);print('PRODUCTION_PROGRAM_BOUND',counts,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);a=p.parse_args();main(a.package)
