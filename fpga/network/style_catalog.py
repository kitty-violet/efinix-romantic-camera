"""Data-only style directory for the standalone S4R1 bitstream.

This module builds user-data Flash contents; it never programs a board. The
FPGA configuration image occupies [0, CATALOG_BASE) and is never overwritten.
The runtime reuses the active compute/weight windows and one CNN. Capacity and
the S4R1 ABI, not the number of newly installed styles, determine compatibility.
"""
from pathlib import Path
import hashlib,json,struct,zlib

FLASH_BYTES=8*1024*1024
CATALOG_BASE=0x340000
MAX_STYLES=8
HEADER_BYTES=64
ENTRY_BYTES=64
DIRECTORY_BYTES=HEADER_BYTES+MAX_STYLES*ENTRY_BYTES
COMPUTE_BYTES=418944
COEFFICIENT_BYTES=20480
DMA_BYTES=132432
MAGIC=0x31433153  # S4R1C2, little endian
ABI=0x00020755
ACTIVE_COMPUTE=0xe1000000
ACTIVE_DMA=0xe1100000
ACTIVE_COEFFICIENT=0xe1200000
RESIDENT_BASE=0xe1800000
RESIDENT_STRIDE=0x100000
RESIDENT_COEFFICIENT=0xc0000

def crc(data):return zlib.crc32(data)&0xffffffff
def align(n,a=4096):return (n+a-1)//a*a
def words(data):return struct.unpack('<'+'I'*(len(data)//4),data)

def build(styles,dma,isp,default_slot=0):
    if not 1<=len(styles)<=6:raise ValueError('style count')
    if not 0<=default_slot<len(styles):raise ValueError('default slot')
    if len(dma)!=DMA_BYTES or len(isp)!=768:raise ValueError('shared payload size')
    data=bytearray(b'\xff'*align(DIRECTORY_BYTES))
    def append(raw):
        pos=align(len(data));data.extend(b'\xff'*(pos-len(data)));data.extend(raw)
        if CATALOG_BASE+len(data)>FLASH_BYTES:raise ValueError('Flash capacity exceeded')
        return CATALOG_BASE+pos
    dma_address=append(dma);isp_address=append(isp)
    entries=[];tags=set()
    for i,s in enumerate(styles):
        compute,coeff=s['compute'],s['coefficients']
        tag=s['tag']
        if type(tag)is not int or not 0<tag<65536 or tag in tags:raise ValueError('unique nonzero tag')
        tags.add(tag)
        if len(compute)!=COMPUTE_BYTES or len(coeff)!=COEFFICIENT_BYTES:raise ValueError('S4R1 payload size')
        if any(coeff[17792:]):raise ValueError('coefficient padding')
        ca=append(compute);wa=append(coeff)
        e=[1,tag,ca,len(compute),crc(compute),wa,len(coeff),crc(coeff),RESIDENT_BASE+i*RESIDENT_STRIDE,0,0,0,0,0,0,0]
        raw=struct.pack('<16I',*e);off=HEADER_BYTES+i*ENTRY_BYTES;data[off:off+ENTRY_BYTES]=raw
        entries.append(dict(slot=i,name=s['name'],tag=tag,compute_sha256=hashlib.sha256(compute).hexdigest(),coefficient_sha256=hashlib.sha256(coeff).hexdigest(),words=e))
    # Unused entries are zero; they cannot accidentally become active slots.
    data[HEADER_BYTES+len(styles)*ENTRY_BYTES:DIRECTORY_BYTES]=bytes((MAX_STYLES-len(styles))*ENTRY_BYTES)
    header=[MAGIC,ABI,len(styles),default_slot,DIRECTORY_BYTES,0,CATALOG_BASE+len(data),dma_address,len(dma),crc(dma),isp_address,len(isp),crc(isp),0,0,0]
    data[:HEADER_BYTES]=struct.pack('<16I',*header)
    header[5]=crc(data[:DIRECTORY_BYTES]);data[:HEADER_BYTES]=struct.pack('<16I',*header)
    validate(bytes(data))
    return bytes(data),dict(format='S1C1',abi=ABI,default_slot=default_slot,styles=entries,flash_offset=CATALOG_BASE,flash_end=header[6],directory_crc32=f'{header[5]:08x}',image_sha256=hashlib.sha256(data).hexdigest(),hardware_access=False)

def validate(data):
    if len(data)<DIRECTORY_BYTES:raise ValueError('truncated directory')
    h=words(data[:HEADER_BYTES])
    if h[0]!=MAGIC or h[1]!=ABI or h[4]!=DIRECTORY_BYTES:raise ValueError('header/ABI')
    if not 1<=h[2]<=MAX_STYLES or h[3]>=h[2]:raise ValueError('slot limits')
    if h[6]!=CATALOG_BASE+len(data) or h[6]>FLASH_BYTES:raise ValueError('Flash bounds')
    if any(h[13:]):raise ValueError('header reserved')
    directory=bytearray(data[:DIRECTORY_BYTES]);directory[20:24]=bytes(4)
    if crc(directory)!=h[5]:raise ValueError('directory CRC')
    regions=[];next_address=CATALOG_BASE+align(DIRECTORY_BYTES)
    def region(addr,size,expected,size_required):
        nonlocal next_address
        if size!=size_required or addr%4096 or addr<CATALOG_BASE+align(DIRECTORY_BYTES) or addr+size>h[6]:raise ValueError('payload bounds/size')
        if addr!=next_address:raise ValueError('noncanonical payload extent')
        next_address=align(addr+size)
        if any(addr<b+n and b<addr+size for b,n in regions):raise ValueError('overlapping payload')
        regions.append((addr,size));p=data[addr-CATALOG_BASE:addr-CATALOG_BASE+size]
        if crc(p)!=expected:raise ValueError('payload CRC')
        return p
    region(h[7],h[8],h[9],DMA_BYTES);region(h[10],h[11],h[12],768)
    tags=set()
    for i in range(MAX_STYLES):
        e=words(data[HEADER_BYTES+i*ENTRY_BYTES:HEADER_BYTES+(i+1)*ENTRY_BYTES])
        if i>=h[2]:
            if any(e):raise ValueError('unused entry')
            continue
        if e[0]!=1 or not 0<e[1]<65536 or e[1]in tags or e[8]!=RESIDENT_BASE+i*RESIDENT_STRIDE or any(e[9:]):raise ValueError('entry ABI/tag/address')
        tags.add(e[1]);region(e[2],e[3],e[4],COMPUTE_BYTES)
        coeff=region(e[5],e[6],e[7],COEFFICIENT_BYTES)
        if any(coeff[17792:]):raise ValueError('coefficient padding')
    if regions[-1][0]+regions[-1][1]!=h[6]:raise ValueError('trailing payload')
    return h
