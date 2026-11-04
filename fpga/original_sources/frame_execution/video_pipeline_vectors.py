"""Diagnostic 1080p source whose exact area reduction equals the CNN fixture.
Only the camera source is injected; all intermediate/output images are checkers.
"""
import json,sys
from pathlib import Path
from run_units import HERE,sha
sys.path.insert(0,str(HERE.parent/'video_scaler'))
from fixed_ratio_resize import half_pixel_tap

def make_video_vectors(vectors,root):
    root.mkdir(exist_ok=False)
    manifest=json.loads((vectors/'frame_manifest.json').read_text())
    source=bytes(b^128 for b in (vectors/'input.bin').read_bytes())
    assert len(source)==720*1280*3
    camera=bytearray([128])*(1080*1920*3)
    for y in range(720):
        cy=(y//2)*3+(y%2)*2
        for x in range(1280):
            cx=(x//2)*3+(x%2)*2
            for c in range(3):
                target=source[(y*1280+x)*3+c]
                corner=(9*target-5*128+2)//4
                assert 0<=corner<=255
                camera[(cy*1920+cx)*3+c]=corner
                # The other five weight units are the 128-valued edges/center.
                assert (4*corner+5*128+4)//9==target
    (root/'camera1080.bin').write_bytes(camera)
    offset=next(n['offset'] for n in manifest['nodes'] if n['id']=='rgb_head')
    with (vectors/'golden.bin').open('rb') as f:
        f.seek(offset);gold=f.read(720*1280*8)
    styled=bytearray(720*1280*3)
    for pixel in range(720*1280):
        for c in range(3):
            styled[pixel*3+c]=source[pixel*3+c] if pixel%1280<640 else gold[pixel*8+c]^128
    xtaps=[half_pixel_tap(x,1280,1920) for x in range(1920)]
    ytaps=[half_pixel_tap(y,720,1080) for y in range(1080)]
    horizontal=bytearray(720*1920*3)
    for y in range(720):
        for x,tap in enumerate(xtaps):
            w=int(tap.weight1*6);assert tap.weight1*6==w
            a=(y*1280+tap.index0)*3;b=(y*1280+tap.index1)*3;out=(y*1920+x)*3
            for c in range(3):horizontal[out+c]=(styled[a+c]*(6-w)+styled[b+c]*w+3)//6
    hdmi=bytearray(1080*1920*3)
    for y,tap in enumerate(ytaps):
        w=int(tap.weight1*6);assert tap.weight1*6==w
        a=tap.index0*5760;b=tap.index1*5760;out=y*5760
        for x in range(5760):hdmi[out+x]=(horizontal[a+x]*(6-w)+horizontal[b+x]*w+3)//6
    (root/'hdmi1080.bin').write_bytes(hdmi)
    result=dict(status='VIDEO_FIXTURES_ONLY_NOT_RTL_PASS',source_reconstruction_pixels=720*1280,
        input_sha256=sha(vectors/'input.bin'),golden_sha256=sha(vectors/'golden.bin'),
        camera_sha256=sha(root/'camera1080.bin'),hdmi_sha256=sha(root/'hdmi1080.bin'),
        generator_sha256=sha(__file__),resize_reference_sha256=sha(HERE.parent/'video_scaler/fixed_ratio_resize.py'))
    (root/'result.json').write_text(json.dumps(result,indent=2))
    return result
