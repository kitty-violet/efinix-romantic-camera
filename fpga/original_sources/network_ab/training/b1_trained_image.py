"""Real B1 coefficients in the existing 17792-byte resident layout.

The shared legacy codec only admits synthetic fixtures. Keep it untouched and
validate the trained flags and three previously reserved residual records here.
The RAM addresses, layer descriptors and word striping remain unchanged.
"""
import struct
import sys
import zlib
from pathlib import Path

sys.path.insert(0,str(Path(__file__).parents[1]))
from coefficient_layout import (HEADER, SECTION_ENTRY, IMAGE_FLAGS, Parameters,
                                build_layout, encode_parameters, decode_image)
from network_ab.model_contract import compile_contract, load_contract

RESIDUAL_MAGIC = 0x31535152  # RQS1


def checksums(image, layout, trained):
    for index, section in enumerate(layout['sections']):
        entry=list(SECTION_ENTRY.unpack_from(image,HEADER.size+index*SECTION_ENTRY.size))
        part=image[section['offset']:section['offset']+section['length']]
        entry[4]=zlib.crc32(part)
        if section['name']=='residual_reserved':entry[5]=0 if trained else 1
        SECTION_ENTRY.pack_into(image,HEADER.size+index*SECTION_ENTRY.size,*entry)
    header=list(HEADER.unpack_from(image));header[7]=zlib.crc32(image[layout['sections'][1]['offset']:]);header[8]=0 if trained else IMAGE_FLAGS
    HEADER.pack_into(image,0,*header)


def encode(parameters, residual):
    rows={r['id']:r for r in compile_contract(load_contract())['nodes']}
    if any(p.relu!=(rows[name]['activation']=='relu') for name,p in parameters.items()):
        raise ValueError('activation differs from frozen B1 graph')
    layout=build_layout();image=bytearray(encode_parameters(parameters))
    section=next(s for s in layout['sections'] if s['name']=='residual_reserved')
    if list(residual)!=['res1_add','res2_add','res3_add']:raise ValueError('all three residuals required')
    for i,(name,r) in enumerate(residual.items()):
        if type(r['main_scaled']) is not bool:raise ValueError('residual branch')
        if not 0<int(r['multiplier'])<131072 or not 0<int(r['post_multiplier'])<131072:raise ValueError('Q17 range')
        struct.pack_into('<8I32x',image,section['offset']+64*i,RESIDUAL_MAGIC,1,i,int(r['main_scaled']),int(r['multiplier']),int(r['post_multiplier']),1,0)
    checksums(image,layout,True)
    return bytes(image)


def decode(image):
    layout=build_layout()
    if len(image)==20480:
        if any(image[17792:]):raise ValueError('nonzero transport padding')
        image=image[:17792]
    if len(image)!=layout['total_bytes']:raise ValueError('image length')
    h=HEADER.unpack_from(image)
    if h[8]!=0:raise ValueError('synthetic/unresolved flags not a trained image')
    if h[7]!=zlib.crc32(image[layout['sections'][1]['offset']:]):raise ValueError('image build CRC')
    for i,s in enumerate(layout['sections']):
        entry=SECTION_ENTRY.unpack_from(image,HEADER.size+i*SECTION_ENTRY.size)
        part=image[s['offset']:s['offset']+s['length']]
        if entry!=(s['kind'],s['offset'],s['length'],64,zlib.crc32(part),0,0,0):raise ValueError('section ABI/CRC')
    section=next(s for s in layout['sections'] if s['name']=='residual_reserved')
    residual={}
    for i in range(3):
        offset=section['offset']+64*i;r=struct.unpack_from('<8I',image,offset)
        if r[:3]!=(RESIDUAL_MAGIC,1,i) or r[3] not in (0,1) or r[6:]!=(1,0) or any(image[offset+32:offset+64]):raise ValueError('residual record')
        if not 0<r[4]<131072 or not 0<r[5]<131072:raise ValueError('residual multiplier')
        residual[f'res{i+1}_add']=dict(main_scaled=bool(r[3]),multiplier=r[4],post_multiplier=r[5])
    # Reuse strict legacy structural/padding checks only after validating all
    # real semantic fields above. Do not mark the delivered trained image synthetic.
    diagnostic=bytearray(image);diagnostic[section['offset']:section['offset']+section['length']]=bytes(section['length'])
    checksums(diagnostic,layout,False);parameters=decode_image(bytes(diagnostic))
    if encode(parameters,residual)!=image:raise ValueError('noncanonical trained image')
    return parameters,residual


def runtime_from_image(image):
    parameters,residual=decode(image)
    return dict(parameters={name:dict(weights=p.weights,bias=p.bias,multipliers=p.multiplier)
                            for name,p in parameters.items()},residual_encoding=residual)
