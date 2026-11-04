"""Freeze bounded offline JTAG payloads; no adapter, board or OpenOCD access."""
import argparse
import hashlib
import json
import shutil
import sys
import zlib
from pathlib import Path
from run_units import ROOT, HERE, sha
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'network_ab'))
from network_ab.coefficient_layout import decode_image


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True,type=Path)
    p.add_argument('--vectors',required=True,type=Path)
    p.add_argument('--descriptors',required=True,type=Path)
    p.add_argument('--boot-image',required=True,type=Path)
    p.add_argument('--tag',type=lambda x:int(x,0),default=0xC321)
    p.add_argument('--frames',type=int,default=3)
    p.add_argument('--diagnostic',action='store_true',help='Explicitly allow the existing synthetic verification weights')
    a=p.parse_args();root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    assert 0<a.tag<=65535 and 1<=a.frames<=16
    vectors=a.vectors.resolve();desc=a.descriptors.resolve();boot=a.boot_image.resolve()
    image=(vectors/'coefficients.bin').read_bytes()
    layout=json.loads((vectors/'coefficient_layout.json').read_text())
    assert sha(vectors/'coefficients.bin')==layout['sha256']
    decode_image(image,expected_sha256=layout['sha256'],require_complete_network=not a.diagnostic)
    assert len(image)==17792
    padded=image+bytes(20480-len(image))
    words=b''.join(int(v,16).to_bytes(8,'little') for v in (vectors/'load_words.hex').read_text().split())
    assert words==padded
    descriptor_meta=json.loads((desc/'result.json').read_text())
    contract=ROOT/'network_ab/model_b_contract.json'
    assert sha(contract)==descriptor_meta['network_contract_sha256']
    boot_meta=json.loads((boot/'boot.json').read_text())
    assert sha(boot/'boot.bin')==boot_meta['sha256']
    root.mkdir(parents=True)
    (root/'coefficients-padded.bin').write_bytes(padded)
    payloads=[]
    for name,cpu,physical,size in (('compute',0xE1000000,0x04000000,8789*64),('dma',0xE1100000,0x04100000,10440*16)):
        src=desc/(name+'.bin');data=src.read_bytes()
        assert len(data)==size and sha(src)==descriptor_meta['streams'][name]['sha256']
        shutil.copy2(src,root/src.name)
        payloads.append(dict(file=src.name,cpu_address=cpu,physical_address=physical,bytes=size,crc32=zlib.crc32(data),sha256=sha(src)))
    payloads.append(dict(file='coefficients-padded.bin',cpu_address=0xE1200000,physical_address=0x04200000,
                         bytes=len(padded),crc32=zlib.crc32(padded),sha256=hashlib.sha256(padded).hexdigest()))
    for r in payloads:
        assert 0xE1000000<=r['cpu_address'] and r['cpu_address']+r['bytes']<=0xE2000000
        assert r['physical_address']==r['cpu_address']-0xE1000000+0x04000000
    mailbox=[dict(address=0xF9000F08,data=a.tag),dict(address=0xF9000F0C,data=zlib.crc32(padded)),
             dict(address=0xF9000F10,data=payloads[0]['crc32']),dict(address=0xF9000F14,data=payloads[1]['crc32']),
             dict(address=0xF9000F18,data=a.frames),dict(address=0xF9000F00,data=0x434E4E31)]
    manifest=dict(status='OFFLINE_PAYLOAD_PACKAGED_NOT_UPLOADED',payloads=payloads,mailbox=mailbox,
                  total_upload_bytes=sum(r['bytes'] for r in payloads),boot_sha256=boot_meta['sha256'],
                  network_contract_sha256=sha(contract),diagnostic_weights=a.diagnostic,frame_budget=a.frames,
                  entry='Wait mailbox phase=1; halt via existing CPU JTAG; upload+readback; write command last; resume CPU',
                  guards=['fixed E1 source window','non-overlapping bounded payloads','host semantic validation','coefficient CRC32 and RAM readback in real CPU','descriptor CRC before frame publication','bounded frame count and runtime watchdog'],
                  board_release=False,full_board_measurement=False)
    (root/'payload.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()
