"""Versioned host container; legacy CNN bytes stay unchanged when extracted.

This is an isolated ABI proposal, not a change to the shared v1 UART loader.
Unknown sections/versions/graphs fail closed; never silently skip style stages.
"""
import hashlib, struct, zlib

MAGIC=b'STYPACK2'
HEADER=192
LIMIT=20480
CNN=1
COLOUR=2
DIAGNOSTIC=0x80000000
SUPPORTED_CAPS=DIAGNOSTIC | 3


def build(style_id, graph_hash, sections, diagnostic=False):
    if not 0<style_id<2**32 or len(graph_hash)!=64:raise ValueError('style id/hash')
    if not sections or len(sections)>8 or len({k for k,v in sections})!=len(sections):raise ValueError('section directory')
    payload=bytearray();directory=bytearray(128);caps=DIAGNOSTIC if diagnostic else 0
    for i,(kind,blob) in enumerate(sections):
        if kind not in (CNN,COLOUR):raise ValueError('unsupported stage')
        if kind==CNN and len(blob)!=17792:raise ValueError('legacy CNN payload must be 17792 bytes')
        if kind==COLOUR:parse_colour(blob)
        while len(payload)%64:payload.append(0)
        offset=HEADER+len(payload);payload.extend(blob);caps|=1<<(kind-1)
        struct.pack_into('<HHIII',directory,16*i,kind,1,offset,len(blob),zlib.crc32(blob))
    if not diagnostic and not caps&1:raise ValueError('production pack requires real CNN section')
    while len(payload)%64:payload.append(0)
    total=HEADER+len(payload)
    if total>LIMIT:raise ValueError('style package exceeds 20KiB')
    header=bytearray(struct.pack('<8sHHIII32sII',MAGIC,1,HEADER,total,style_id,caps,
                                 bytes.fromhex(graph_hash),zlib.crc32(payload),0)+directory)
    struct.pack_into('<I',header,60,zlib.crc32(header))
    return bytes(header+payload)


def parse(blob, expected_graph, allow_diagnostic=False):
    if len(blob)<HEADER or len(blob)>LIMIT:raise ValueError('length')
    magic,version,header,total,style_id,caps,graph,pcrc,hcrc=struct.unpack_from('<8sHHIII32sII',blob)
    if (magic,version,header,total)!=(MAGIC,1,HEADER,len(blob)):raise ValueError('header ABI')
    if not style_id or caps&~SUPPORTED_CAPS:raise ValueError('id/capability')
    if graph.hex()!=expected_graph:raise ValueError('wrong network graph')
    if caps&DIAGNOSTIC and not allow_diagnostic:raise ValueError('diagnostic is not deployable full style')
    checked=bytearray(blob[:HEADER]);struct.pack_into('<I',checked,60,0)
    if zlib.crc32(checked)!=hcrc or zlib.crc32(blob[HEADER:])!=pcrc:raise ValueError('CRC')
    entries={};ranges=[]
    for i in range(8):
        kind,rev,offset,length,crc=struct.unpack_from('<HHIII',blob,64+16*i)
        if kind==0:
            if any((rev,offset,length,crc)):raise ValueError('invalid empty directory')
            continue
        if kind not in (CNN,COLOUR) or rev!=1 or kind in entries:raise ValueError('stage ABI')
        if offset<HEADER or offset%64 or offset+length>len(blob):raise ValueError('section bounds')
        if any(offset<b and offset+length>a for a,b in ranges):raise ValueError('overlapping sections')
        part=blob[offset:offset+length]
        if zlib.crc32(part)!=crc:raise ValueError('section CRC')
        if kind==CNN and length!=17792:raise ValueError('legacy CNN length')
        if kind==COLOUR:parse_colour(part)
        entries[kind]=part;ranges.append((offset,offset+length))
    actual_caps=sum(1<<(k-1) for k in entries)
    if (caps&3)!=actual_caps or not entries:raise ValueError('capability mismatch')
    if not caps&DIAGNOSTIC and CNN not in entries:raise ValueError('missing CNN')
    return dict(style_id=style_id,graph_sha256=graph.hex(),diagnostic=bool(caps&DIAGNOSTIC),sections=entries)


def colour_section(rgb_bytes,bypass=False):
    if len(rgb_bytes)!=729*3:raise ValueError('colour table size')
    return struct.pack('<HHI',1,9,int(bypass))+rgb_bytes


def parse_colour(blob):
    if len(blob)!=8+729*3:raise ValueError('colour length')
    version,grid,flags=struct.unpack_from('<HHI',blob)
    if version!=1 or grid!=9 or flags&~1:raise ValueError('colour ABI')
    return dict(bypass=bool(flags&1),table=blob[8:])


def legacy_transport(parsed):
    """Existing loader still receives its exact 17792B image + zero padding."""
    if CNN not in parsed['sections']:raise ValueError('no CNN')
    return parsed['sections'][CNN]+bytes(LIMIT-17792)


def colour_configuration(parsed):
    """A legacy CNN-only style explicitly requests exact RGB bypass."""
    if COLOUR not in parsed['sections']:
        return dict(bypass=True,table=bytes(729*3))
    return parse_colour(parsed['sections'][COLOUR])
