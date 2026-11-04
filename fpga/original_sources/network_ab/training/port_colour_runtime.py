"""Runtime style-bank replacement for the static colour-only probe."""
from pathlib import Path
import argparse,json,shutil,struct,sys,zlib
import numpy as np
from .block_video_runtime import STILLS
from .port_colour_lut import interpolate
from .runtime import file_hash,write_json
from .build_region_rtl import BIN,VROOT,run
from . import style_pack as pack

OUT=STILLS.parent/'fpga-style-runtime-20261005-v2'
PRIOR=STILLS.parent/'fpga-style-port-20261005-v2'


def prepare():
    OUT.mkdir(exist_ok=False)
    source=OUT/'source_code';source.mkdir()
    for f in [Path(__file__),Path(__file__).with_name('style_pack.py'),
              Path(__file__).with_name('port_colour_lut.py'),
              Path(__file__).with_name('fpga_style_port')/'style_colour_runtime.sv',
              Path(__file__).with_name('fpga_style_port')/'colour_runtime.cpp']:
        shutil.copy2(f,source/f.name)
        if f.suffix in ('.sv','.cpp'):shutil.copy2(f,OUT/f.name)
    table=np.load(PRIOR/'tone_lut.npy');table.tofile(OUT/'table.rgb')
    second=table[:,[1,2,0]].copy();second.tofile(OUT/'table2.rgb')
    shutil.copy2(PRIOR/'input.rgb',OUT/'input.rgb');shutil.copy2(PRIOR/'gold.rgb',OUT/'gold.rgb')
    pixels=np.fromfile(OUT/'input.rgb',np.uint8).reshape(-1,3)
    interpolate(pixels,second).tofile(OUT/'gold2.rgb')
    contract=file_hash(Path('E:/fpga/network_ab/model_b_contract.json'))
    section=pack.colour_section(table.tobytes())
    current=pack.build(1,contract,[(pack.COLOUR,section)],diagnostic=True)
    diagnostic=pack.build(2,contract,[(pack.COLOUR,pack.colour_section(second.tobytes()))],diagnostic=True)
    (OUT/'current_colour_only.stylepack').write_bytes(current)
    (OUT/'diagnostic_colour_rotation.stylepack').write_bytes(diagnostic)
    parsed=pack.parse(current,contract,allow_diagnostic=True)
    assert pack.parse_colour(parsed['sections'][pack.COLOUR])['table']==table.tobytes()
    # This synthetic buffer verifies transport compatibility, not trained weights.
    legacy=bytes((i*31+7)%256 for i in range(17792))
    full=pack.build(7,contract,[(pack.CNN,legacy),(pack.COLOUR,section)],diagnostic=True)
    unpacked=pack.parse(full,contract,allow_diagnostic=True)
    transport=pack.legacy_transport(unpacked)
    assert transport[:17792]==legacy and transport[17792:]==bytes(2688) and len(transport)==20480
    legacy_only=pack.parse(pack.build(8,contract,[(pack.CNN,legacy)],diagnostic=True),contract,True)
    assert pack.colour_configuration(legacy_only)['bypass']
    rejected=[]
    def reject(name,action):
        try:action()
        except ValueError:rejected.append(name)
        else:raise AssertionError(name)
    reject('diagnostic full-style activation',lambda:pack.parse(current,contract))
    reject('wrong graph',lambda:pack.parse(current,'00'*32,True))
    reject('truncated packet',lambda:pack.parse(current[:-1],contract,True))
    corrupt=bytearray(current);corrupt[-12]^=1
    reject('payload CRC',lambda:pack.parse(corrupt,contract,True))
    corrupt=bytearray(current);corrupt[16]^=1
    reject('header CRC',lambda:pack.parse(corrupt,contract,True))
    reject('duplicate sections',lambda:pack.build(1,contract,[(pack.COLOUR,section)]*2,True))
    reject('unknown stage',lambda:pack.build(1,contract,[(33,b'x')],True))
    reject('wrong LUT size',lambda:pack.colour_section(bytes(3)))
    reject('wrong CNN size',lambda:pack.build(1,contract,[(pack.CNN,bytes(20480))],True))
    reject('production without CNN',lambda:pack.build(1,contract,[(pack.COLOUR,section)]))
    # CRC-valid unknown schema must still fail rather than silently drop a stage.
    altered=bytearray(current);struct.pack_into('<H',altered,66,9);struct.pack_into('<I',altered,60,0);struct.pack_into('<I',altered,60,zlib.crc32(altered[:192]))
    reject('CRC-valid unknown section revision',lambda:pack.parse(altered,contract,True))
    write_json(OUT/'pack_verification.json',dict(rejected_cases=rejected,current_colour_only_bytes=len(current),
        synthetic_full_container_bytes=len(full),limit=20480,legacy_extracted_exact=True,
        old_loader_receives_zero_padding=True,legacy_without_colour_uses_exact_bypass=True,
        full_trained_package=False,flash_offsets_assigned=False))
    template=(PRIOR/'style_colour_lut.xml').read_text().replace('style_colour_lut','style_colour_runtime')
    (OUT/'style_colour_runtime.xml').write_text(template,encoding='utf-8')
    (OUT/'style_colour_runtime.sdc').write_text('create_clock -name core_clk -period 6.667 [get_ports clk]\n',encoding='ascii')
    write_json(OUT/'inputs.json',dict(contract_sha256=contract,
        source={p.name:file_hash(p) for p in source.iterdir()},
        tables={p.name:file_hash(p) for p in [OUT/'table.rgb',OUT/'table2.rgb']},
        approved_reference=str(STILLS.parent/'gentle-global-boundaries-20261005-v3'),
        changed_shared_hardware=False,full_style=False,scope='runtime-loadable fixed colour stage'))
    print('RUNTIME_PREPARED',OUT,'pack tests=',len(rejected),flush=True)


def simulate():
    obj=OUT/'obj';obj.mkdir(exist_ok=True)
    elapsed=run([BIN/'verilator_bin.exe','--cc','--exe','--top-module','style_colour_runtime','-Wno-fatal',
        '--Mdir',obj.as_posix(),(OUT/'style_colour_runtime.sv').as_posix(),(OUT/'colour_runtime.cpp').as_posix()],OUT,OUT/'compile.log')
    make=[BIN/'mingw32-make.exe','-f','Vstyle_colour_runtime.mk','-j2',
          f'CXX={(BIN/"g++.exe").as_posix()}',f'LINK={(BIN/"g++.exe").as_posix()}',f'AR={(BIN/"ar.exe").as_posix()}',
          'PERL=E:/GIT/Git/usr/bin/perl.exe','SHELL=E:/GIT/Git/usr/bin/sh.exe',
          f'PYTHON3={Path(sys.executable).as_posix()}',f'VERILATOR_ROOT={VROOT.as_posix()}',
          'OPT_FAST=-O2','OPT_GLOBAL=-O2']
    elapsed+=run(make,obj,OUT/'build.log',180)
    binary=obj/'Vstyle_colour_runtime.exe'
    elapsed+=run([binary],OUT,OUT/'simulation.log',120)
    body=(OUT/'simulation.log').read_text();assert 'RUNTIME_STYLE_PASS' in body
    write_json(OUT/'rtl_result.json',dict(status='RUNTIME_COLOUR_STAGE_PASS',log=body,seconds=elapsed,
        rtl_sha256=file_hash(OUT/'style_colour_runtime.sv'),binary_sha256=file_hash(binary),
        full_style=False,whole_board_fit=False,physical_timing=False))
    print(body,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','simulate']);a=p.parse_args()
    prepare() if a.mode=='prepare' else simulate()
