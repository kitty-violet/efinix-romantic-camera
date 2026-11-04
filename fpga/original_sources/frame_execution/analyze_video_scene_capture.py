"""Analyze confirmed frozen samples; never substitute sparse data for a full frame."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path
from PIL import Image
from video_input_isp_control import table

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def stats(pixels):
    return dict(pixels=len(pixels), mean_rgb=[round(statistics.mean(p[c] for p in pixels),4) for c in range(3)],
                min_rgb=[min(p[c] for p in pixels) for c in range(3)],
                max_rgb=[max(p[c] for p in pixels) for c in range(3)],
                unique_rgb=len(set(map(tuple,pixels))),
                clipped255_rgb=[sum(p[c] == 255 for p in pixels) for c in range(3)],
                zero_rgb=[sum(p[c] == 0 for p in pixels) for c in range(3)])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    a=parser.parse_args();root=a.root.resolve();out=root/'user-framed-analysis1.json'
    assert not out.exists()
    sparse=json.loads((root/'user-framed-sparse1.json').read_text())
    assert sparse['status']=='SPARSE_INPUT_READ_ONLY_PASS' and sparse['control']['errors']==0
    assert sparse['pixels_read']==3600 and sparse['path_before']['inspect_bank']==sparse['path_after']['inspect_bank']==0
    im=Image.open(root/'user-framed-sparse1.ppm').convert('RGB')
    assert im.size==(80,45) and sha(root/'user-framed-sparse1.ppm')==sparse['image_sha256']
    data=table(1.6,(1.1431,.7730,1.4135),0)
    assert hashlib.sha256(data).hexdigest()=='36df26ca1a449e55fd20036d62b80d3e46a7d6dc5a4fdf2e634c3a83eceb8eb0'
    applied=json.loads((root/'contrast-isp1.json').read_text())
    assert applied['status']=='INPUT_ISP_CRC_AND_ALL_768_BYTES_VERIFIED_STOPPED'
    assert applied['readbacks']==[[data[c*256+i] for c in range(3)] for i in range(256)]
    samples={};images=[]
    for name in ['user-framed-lock1','user-framed-lower1']:
        report=json.loads((root/(name+'.json')).read_text())
        info=report['frame_prefix'];p=Path(info['file'])
        assert report['control']['errors']==0 and sha(p)==info['sha256']
        assert info['before']['inspect_bank']==info['after']['inspect_bank']==0
        y,rem=divmod(info['offset'],1280*3);x=rem//3
        raw=bytes(v^128 for v in p.read_bytes()); pixels=list(zip(raw[::3],raw[1::3],raw[2::3]))
        assert len(pixels)==1024 and x==128
        # The readout view uses little-endian RGB q8, not a screen photograph.
        line=Image.frombytes('RGB',(1024,1),raw).resize((1024,48),Image.Resampling.NEAREST)
        images.append((y,line))
        runs=[];start=0
        for i in range(1,len(pixels)+1):
            if i==len(pixels) or pixels[i]!=pixels[start]:
                runs.append(dict(x_start=x+start,x_end=x+i-1,length=i-start,rgb=pixels[start]));start=i
        samples[str(y)]=dict(offset=info['offset'],bytes=info['bytes'],sha256=sha(p),
            complete_frame=False,largest_constant_run=max(runs,key=lambda r:r['length']),
            regions={str(x+k):stats(pixels[k:k+128]) for k in range(0,1024,128)})
    canvas=Image.new('RGB',(1024,96))
    for i,(y,line) in enumerate(images):canvas.paste(line,(0,i*48))
    canvas.save(root/'user-framed-lines1.png')
    result=dict(status='CONFIRMED_SCENE_SAMPLES_ANALYZED_NOT_IMAGE_ACCEPTANCE',hardware_access=False,
        source_size=[1280,720],sparse_size=[80,45],complete_frame=False,
        sparse_stats=stats(list(im.getdata())),rows=samples,
        isp_crc32='dbe557c4',isp_max_rgb=[max(data[c*256:(c+1)*256]) for c in range(3)],
        isp_first_input_mapping_to_255=[next((i for i in range(256) if data[c*256+i]==255),None) for c in range(3)],
        conclusions=['A constant highlight plateau exists before HDMI; lost detail cannot be recovered by gamma.',
                     'Current WB LUT clips R/B early; G maximum is 217. Saturated highlights can be magenta.',
                     'Row512 x512..639 is nearly neutral; do not infer global green excess from a screen photograph.',
                     'Dark sampled pixels are nonzero and vary; capture does not prove their original detail is retained.',
                     'No pre-LUT RAW sample: cannot separate sensor saturation, optics, demosaic and LUT by these samples alone.'],
        evidence={str(p):sha(p) for p in [root/'user-framed-sparse1.json',root/'user-framed-lock1.json',
                  root/'user-framed-lower1.json',root/'contrast-isp1.json',Path(__file__)]})
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','isp_max_rgb','isp_first_input_mapping_to_255']}))

if __name__=='__main__':main()
