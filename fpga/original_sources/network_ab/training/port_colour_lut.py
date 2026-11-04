"""First isolated fixed-point colour stage; not a full FPGA style deployment."""
from pathlib import Path
import argparse, json, shutil, time, os, subprocess, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .block_video_runtime import STILLS, load_frozen
from .runtime import file_hash, write_json
from .build_region_rtl import BIN, VROOT, run

OUT=STILLS.parent/'fpga-style-port-20261005-v2'


def interpolate(rgb, table):
    shape=rgb.shape;x=rgb.reshape(-1,3).astype(np.int32)
    low=x>>5;f=x&31;order=np.argsort(-f,axis=1,kind='stable')
    fractions=np.take_along_axis(f,order,axis=1)
    weights=np.stack((32-fractions[:,0],fractions[:,0]-fractions[:,1],
                      fractions[:,1]-fractions[:,2],fractions[:,2]),axis=1)
    strides=np.array([81,9,1],np.int32)[order]
    base=low@np.array([81,9,1],np.int32)
    addr=np.stack((base,base+strides[:,0],base+strides[:,:2].sum(1),base+91),1)
    result=((table.reshape(-1,3).astype(np.int32)[addr]*weights[:,:,None]).sum(1)+16)>>5
    assert result.min()>=0 and result.max()<=255
    return result.astype(np.uint8).reshape(shape)


def prepare():
    OUT.mkdir(exist_ok=False)
    reference=STILLS.parent/'gentle-global-boundaries-20261005-v3'
    assert (reference/'manifest.json').exists()
    _,direction,paint=load_frozen()
    adapter=np.load(STILLS/'color_adapter.npz')['lab_chroma_affine']
    grid=np.arange(9,dtype=np.int32)*32
    rgb=np.stack(np.meshgrid(grid,grid,grid,indexing='ij'),-1).clip(0,255).astype(np.uint8).reshape(9,81,3)
    table=paint.tone(rgb,adapter).reshape(729,3)
    np.save(OUT/'tone_lut.npy',table)
    (OUT/'tone_lut.hex').write_text(''.join(f'{int(r):02x}{int(g):02x}{int(b):02x}\n' for r,g,b in table),encoding='ascii')
    source=OUT/'source_code';source.mkdir()
    for f in [Path(__file__),Path(__file__).with_name('fpga_style_port')/'style_colour_lut.sv',Path(__file__).with_name('fpga_style_port')/'colour_lut.cpp']:
        shutil.copy2(f,source/f.name)
    shutil.copy2(source/'style_colour_lut.sv',OUT/'style_colour_lut.sv')
    shutil.copy2(source/'colour_lut.cpp',OUT/'colour_lut.cpp')
    cfg=json.loads((STILLS/'experiment.json').read_text(encoding='utf-8'))
    results=[];pixels=[]
    rng=np.random.default_rng(20261005)
    pixels.append(rng.integers(0,256,(65536,3),dtype=np.uint8))
    # Exhaustive per-channel ramps and all lattice vertices/combinations of ties.
    for c in range(3):
        for anchor in (0,31,32,127,224,255):
            ramp=np.full((256,3),anchor,np.uint8);ramp[:,c]=np.arange(256);pixels.append(ramp)
    pixels.append(rgb.reshape(-1,3))
    board=Image.new('RGB',(840,3*454),'#17202a');draw=ImageDraw.Draw(board);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    for i,(key,box) in enumerate([('val_plaza_090',(0,240,420,660)),('val_stairs_090',(120,300,540,720)),('val_new_hallway_120',(0,0,420,420))]):
        row=next(r for r in cfg['inputs'] if r['id']==key)
        path=Path(row['neural_prediction']['actual_student_prediction'])
        actual=np.array(Image.open(path))
        direct=paint.tone(actual,adapter);fixed=interpolate(actual,table)
        Image.fromarray(fixed).save(OUT/(key+'_integer_tone.png'))
        Image.fromarray(direct).save(OUT/(key+'_float_tone.png'))
        err=np.abs(direct.astype(np.int16)-fixed.astype(np.int16))
        results.append(dict(id=key,input=str(path),input_sha256=file_hash(path),mae=float(err.mean()),p99=float(np.percentile(err,99)),maximum=int(err.max())))
        for col,(a,title) in enumerate([(direct,'原浮点色彩阶段'),(fixed,'FPGA整数查表色彩阶段')]):
            draw.text((420*col+5,454*i+4),title,font=font,fill='white');board.paste(Image.fromarray(a).crop(box),(420*col,454*i+34))
        if i==0:pixels.append(actual.reshape(-1,3))
    board.save(OUT/'colour_stage_comparison.png')
    vectors=np.concatenate(pixels);vectors.tofile(OUT/'input.rgb');interpolate(vectors,table).tofile(OUT/'gold.rgb')
    count=len(direction.sector_samples(4.)[0])
    write_json(OUT/'stage_manifest.json',dict(status='FIRST_COLOUR_STAGE_ONLY',approved_reference=str(reference),
        approved_reference_manifest_sha256=file_hash(reference/'manifest.json'),
        contract_sha256=file_hash(Path('E:/fpga/network_ab/model_b_contract.json')),
        stage='fixed tone/chroma/highlight before direction4',lut_nodes=729,lut_payload_bytes=2187,
        integer='RGB8 tetrahedral interpolation Q5 nonnegative barycentric weights; nearest rounding',
        comparisons=results,verification_vectors=len(vectors),
        unported=['real StrokeColorInputNet FP32 graph differs from B1; INT8 export and scheduler adaptation pending',
                  f'direction4 uses {count} bilinear sample positions and 8 sector moments per native pixel',
                  'multi-scale local contrast and Lab-dependent block/edge processing',
                  'LRASPP source person mask, mask refinement and weighted TV people branch',
                  'gentle-v3 spatial boundary tightening'],
        full_style_rtl=False,whole_board_fit=False,physical_timing=False,uploaded=False))
    template=Path('E:/fpga/model-training-workspace/runs/region-detail-deploy-20261004/map-row-decode-static2/region_row_decode.xml').read_text(encoding='utf-8')
    template=template.replace('<efx:design_file name="region_luma_decode.sv" version="default" library="default" />','').replace('region_row_decode','style_colour_lut')
    (OUT/'style_colour_lut.xml').write_text(template,encoding='utf-8')
    (OUT/'style_colour_lut.sdc').write_text('create_clock -name core_clk -period 6.667 [get_ports clk]\n',encoding='ascii')
    print(json.dumps(dict(output=str(OUT),records=results,vectors=len(vectors)),ensure_ascii=False),flush=True)


def simulate():
    obj=OUT/'obj';obj.mkdir(exist_ok=True)
    args=[BIN/'verilator_bin.exe','--cc','--exe','--top-module','style_colour_lut','-Wno-fatal',
          '--Mdir',obj.as_posix(),(OUT/'style_colour_lut.sv').as_posix(),(OUT/'colour_lut.cpp').as_posix()]
    elapsed=run(args,OUT,OUT/'compile.log')
    make=[BIN/'mingw32-make.exe','-f','Vstyle_colour_lut.mk','-j2',f'CXX={(BIN/"g++.exe").as_posix()}',
          f'LINK={(BIN/"g++.exe").as_posix()}',f'AR={(BIN/"ar.exe").as_posix()}',
          'PERL=E:/GIT/Git/usr/bin/perl.exe','SHELL=E:/GIT/Git/usr/bin/sh.exe',
          f'PYTHON3={Path(sys.executable).as_posix()}',
          f'VERILATOR_ROOT={VROOT.as_posix()}','OPT_FAST=-O2','OPT_GLOBAL=-O2']
    elapsed+=run(make,obj,OUT/'build.log',180)
    binary=obj/'Vstyle_colour_lut.exe'
    elapsed+=run([binary],OUT,OUT/'simulation.log',120)
    body=(OUT/'simulation.log').read_text();assert 'COLOUR_LUT_PASS' in body
    write_json(OUT/'rtl_result.json',dict(status='COLOUR_STAGE_RTL_PASS',log=body,seconds=elapsed,
        rtl_sha256=file_hash(OUT/'style_colour_lut.sv'),lut_sha256=file_hash(OUT/'tone_lut.hex'),
        binary_sha256=file_hash(binary),whole_style=False,physical_timing=False))
    print(body,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','simulate']);a=p.parse_args()
    prepare() if a.mode=='prepare' else simulate()
