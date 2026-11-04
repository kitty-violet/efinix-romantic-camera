"""Read-only R39-medium package verification. Python standard library only."""
from pathlib import Path
import ast
import hashlib
import json
import struct
import sys
import zlib

EXPECTED_SHA='1c97b7c6aa578a64e259f15a4e3c6d2a932ffda9f02ae06ee3b14c48e2a73d90'

def verify(root):
    if not __debug__:raise RuntimeError('Run without -O; verification uses assertions.')
    root=Path(root).resolve()
    read=lambda p:json.loads((root/p).read_text('utf-8'))
    manifest=read('manifest-sha256.json');listed=set()
    for row in manifest['files']:
        p=(root/row['path']).resolve()
        assert root in p.parents and p.is_file(),row['path']
        assert row['path'] not in listed
        listed.add(row['path']);data=p.read_bytes()
        assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256'],row['path']
    assert {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==listed|{'manifest-sha256.json'}
    assert not any('r40' in p.lower() for p in listed)
    a=read('acceptance.json')
    assert a['name']=='像素动画风格' and a['selected_candidate']=='pixel-A-sharp-medium-r39'
    assert a['sharpness']=='medium' and a['native_cell']==[4,4] and a['excluded_R40']
    image=(root/'coefficients-padded.bin').read_bytes();effective=(root/'coefficients.bin').read_bytes()
    assert len(image)==20480 and len(effective)==17792
    assert image[:17792]==effective and not any(image[17792:])
    words=(root/'load_words.hex').read_text('ascii').splitlines()
    assert len(words)==2560 and all(len(w)==16 for w in words)
    assert b''.join(int(w,16).to_bytes(8,'little') for w in words)==image
    assert hashlib.sha256(image).hexdigest()==a['coefficient_sha256']==EXPECTED_SHA
    assert f'{zlib.crc32(image):08x}'==a['crc32_padded']=='2d9aad69'
    layout=read('contract/coefficient_layout.json');h=struct.unpack_from('<8s8I24s',effective)
    assert h[:7]==(b'TI60CFB1',1,17792,24,len(layout['sections']),64,2)
    assert h[8:]==(0,bytes(24))
    assert h[7]==zlib.crc32(effective[layout['sections'][1]['offset']:])
    for i,s in enumerate(layout['sections']):
        data=effective[s['offset']:s['offset']+s['length']]
        assert struct.unpack_from('<8I',effective,64+32*i)==(s['kind'],s['offset'],s['length'],64,zlib.crc32(data),0,0,0)
    residual=read('residual_encoding.json')
    offset=next(s['offset'] for s in layout['sections'] if s['name']=='residual_reserved')
    for i,(name,r) in enumerate(residual.items()):
        assert name==f'res{i+1}_add' and r==dict(main_scaled=False,multiplier=1,post_multiplier=131071)
        assert struct.unpack_from('<8I',effective,offset+64*i)==(0x31535152,1,i,0,1,131071,1,0)
        assert not any(effective[offset+64*i+32:offset+64*i+64])
    counts=read('parameter-count.json')
    assert counts['weights']==sum(o['raw_weights'] for o in layout['operators'])==10256
    assert counts['biases']==sum(o['output_channels'] for o in layout['operators'])==563
    assert counts['network_parameters']==sum(r['weights']+r['biases'] for r in counts['layers'])==10819
    assert counts['q17_channel_multipliers']==563
    v=read('evidence/integer-verification.json')
    assert v['sha256_padded']==EXPECTED_SHA and v['small_oracle_scalar_comparisons']==68800
    assert all(r['max_error']==0 for r in v['full720_qat_vs_decoded_integer_nodes'].values())
    assert hashlib.sha256((root/'selected.pt').read_bytes()).hexdigest()==v['selected_checkpoint_sha256']
    b=read('evidence/selected-video-binding.json');assert b['coefficient_sha256']==EXPECTED_SHA
    audit=read('evidence/package-audit.json')
    assert audit['checkpoint_reexport_byte_identical'] and audit['video_decode_complete']
    clips={}
    for scene in ('corridor','outdoor'):
        r=read(f'evidence/{scene}-report.json');pts=r['pts']
        assert r['frames']==len(pts)==r['unique_raw_frames']==91
        assert all(y>x for x,y in zip(pts,pts[1:]))
        assert r['coefficient_sha256']==EXPECTED_SHA
        video=(root/'previews'/f'{scene}.mp4').read_bytes()
        assert hashlib.sha256(video).hexdigest()==r['video']['sha256']
        assert r['video']['max_pts_error_seconds']<.001 and r['video']['no_frame_drop_or_duplication']
        assert audit['videos'][scene]['frames']==audit['videos'][scene]['unique_frames']==91
        clips[scene]=dict(frames=91,seconds=r['seconds'])
    sample=read('test_vectors/sample.json')
    assert sample['coefficient_sha256']==EXPECTED_SHA and sample['qat_integer_byte_exact']
    for key in ('input_rgb888.bin','expected_rgb888.bin'):
        raw=(root/'test_vectors'/key).read_bytes()
        assert len(raw)==1280*720*3 and hashlib.sha256(raw).hexdigest()==sample['files'][key]['sha256']
    assert sample['strict4_constant'] and not sample['strict8_constant']
    for p in root.rglob('*.py'):ast.parse(p.read_text('utf-8-sig'),filename=str(p))
    return dict(status='PASS',name=a['name'],version='R39 medium',files=len(listed),parameters=10819,
                coefficient_sha256=EXPECTED_SHA,crc32='2d9aad69',clips=clips,vectors_checked=True,board_test=False)

if __name__=='__main__':
    print(json.dumps(verify(sys.argv[1] if len(sys.argv)>1 else Path(__file__).resolve().parent),ensure_ascii=False,indent=2))
