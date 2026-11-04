"""Short real coefficient-store/RQ RTL test; never a full-frame board claim."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from .b1_trained_image import decode
from .runtime import file_hash, write_json


def main(package):
    package=package.resolve();out=package/'rtl-package-check';out.mkdir(exist_ok=False)
    image=(package/'coefficients-padded.bin').read_bytes();parameters,residual=decode(image)
    shutil.copy2(package/'load_words.hex',out/'load_words.hex')
    raw=b''.join(int(line,16).to_bytes(8,'little') for line in (out/'load_words.hex').read_text().splitlines())
    assert raw==image
    rng=np.random.default_rng(20261005);vectors=[]
    def add(value,mult,relu):
        value=max(-(1<<31),min((1<<31)-1,int(value)));product=value*int(mult)
        y=(abs(product)+65536)//131072
        if product<0:y=-y
        y=max(0 if relu else -128,min(127,y))
        vectors.append((value&0xffffffff)|((int(mult)&0xffffffff)<<32)|(int(relu)<<64)|((y&255)<<72))
    for name,p in parameters.items():
        for oc,w in enumerate(p.weights):
            for a in (rng.integers(-128,128,len(w)),np.full(len(w),-128),np.full(len(w),127),np.zeros(len(w))):
                dot=sum(int(x)*int(y) for x,y in zip(a,w));add(dot+p.bias[oc],p.multiplier[oc],p.relu)
            add(-(1<<31),p.multiplier[oc],p.relu);add((1<<31)-1,p.multiplier[oc],p.relu)
    for r in residual.values():
        for x in range(-128,128):
            add(x,r['multiplier'],False);add(x*2,r['post_multiplier'],False)
    assert len(vectors)<=8192
    (out/'rq_vectors.hex').write_text(''.join(f'{v:020x}\n' for v in vectors),encoding='ascii')
    (out/'rq_count.txt').write_text(str(len(vectors)),encoding='ascii')
    project=Path(r'E:\fpga');ef=Path(r'E:\efinity-2026.1.132-windows-x64\efinity\2026.1')
    files=[project/'hw/efinity/prototypes/frame_execution/variants/fixed_source_cursor_shared/coefficient_bank_store_fast.sv',
           project/'hw/efinity/prototypes/cnn_microbench/cnn_weight_bank64.sv',
           project/'hw/efinity/prototypes/frame_execution/variants/fixed_source_cursor_shared/requantize_i32_mul18_q17_pipeline.sv',
           ef/'sim_models/verilog/efx_dsp48.v',Path(__file__).with_name('fpga_style_port')/'tb_b1_trained_package.sv']
    frozen=[]
    for f in files:
        shutil.copy2(f,out/f.name);frozen.append(dict(source=str(f),file=f.name,sha256=file_hash(f)))
    write_json(out/'inputs.json',dict(source=frozen,weights_sha256=file_hash(package/'coefficients-padded.bin'),rq_vectors=len(vectors)))
    sim=Path(r'E:\altera\15.0\modelsim_ase\win32aloem');env=os.environ.copy();env.update(TEMP=str(out),TMP=str(out))
    (out/'run.do').write_text('onerror {quit -code 1}\nonbreak {quit -code 1}\nrun -all\nquit -code 0\n')
    stages=[]
    for name,args in [('library',['vlib.exe','work']),('compile',['vlog.exe','-sv','-timescale','1ns/1ps',*[f.name for f in files]]),
                      ('simulation',['vsim.exe','-c','-wlf',str(out/'simulation.wlf'),'work.tb_b1_trained_package','-do','run.do'])]:
        args[0]=str(sim/args[0]);start=time.perf_counter()
        with (out/(name+'.log')).open('w',encoding='utf-8') as log:
            proc=subprocess.run(args,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        stages.append(dict(stage=name,seconds=time.perf_counter()-start,exit_code=proc.returncode))
        if proc.returncode:raise RuntimeError((out/(name+'.log')).read_text(errors='replace')[-2500:])
    body=(out/'simulation.log').read_text(errors='replace')
    marker=f'PASS TRAINED_B1_STORE_RQ readback_words=2560 fast_words=160 rq={len(vectors)}'
    assert marker in body and 'Errors: 0, Warnings: 0' in body,body[-2500:]
    readback=b''.join(int(line,16).to_bytes(8,'little') for line in (out/'readback.hex').read_text().splitlines())
    assert readback==image;assert decode(readback)==decode(image)
    assert all(file_hash(x['source'])==x['sha256']==file_hash(out/x['file']) for x in frozen)
    write_json(out/'result.json',dict(status='ACTUAL_TRAINED_PACKAGE_RTL_STORE_RQ_PASS',stages=stages,marker=marker,
                                    official_dsp_model=True,readback_bytes=len(readback),full_network_rtl=False,
                                    full_board=False,weights_sha256=file_hash(package/'coefficients-padded.bin')))
    print(marker,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);a=p.parse_args();main(a.package)
