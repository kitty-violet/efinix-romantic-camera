"""Offline official BRAM-content update on a frozen routed implementation.

The base map intentionally remains unchanged (Efinity's documented updater
contract). Release identity is base implementation + verified ROM overlay.
Never opens USB, programs an FPGA, changes constraints, or runs P&R.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from run_units import ROOT, EF, sha
from prepare_video_service_color_exposure import read_rom
from windows_process_tree_safe import stop_launched_tree

BASE = ROOT/'tmp/frame-execution/video-input-bit2'
CANDIDATE = ROOT/'tmp/frame-execution/video-sensor-gain70-candidate1'
TARGET = 'camera_checked/config_rom/ROM__D$2'
UPDATER = EF/'bin/efx_bram_edit.exe'
GEOMETRY = ['outflow/ti60f225_oob'+s for s in ('.map.v','.place','.troutingtraces','.vdb')]
GEOMETRY += ['ti60f225_oob.sdc','ti60f225_oob.xml','native_source.v',
             'outflow/ti60f225_oob.raminfo.pb','outflow/ti60f225_oob.peri.xml']


def decode_read(text):
    chunks = re.split(r'\*+ Memory "([^"]+)" \*+', text)
    result = {}
    for i in range(1, len(chunks), 2):
        name, body = chunks[i:i+2]
        bits = ''.join(x.strip() for x in body.splitlines() if re.fullmatch('[01]+', x.strip()))
        assert name not in result and bits
        result[name] = bits
    return result


def execute(args, root, name, timeout=60):
    env = os.environ.copy(); env['EFINITY_HOME'] = str(EF)
    env['PATH'] = str(EF/'bin')+';'+env['PATH']
    start = time.monotonic()
    with (root/name).open('xb') as log:
        proc = subprocess.Popen(list(map(str,args)), cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        try: rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            (root/(name+'.timeout.json')).write_text(json.dumps(stop_launched_tree(proc),indent=2))
            raise
    text = (root/name).read_text(errors='replace')
    assert rc == 0 and not re.search(r'(?m)^\s*ERROR\s*:',text), (name,rc,text[-2000:])
    return text, dict(command=list(map(str,args)),seconds=time.monotonic()-start,exit_code=rc,log_sha256=sha(root/name))


def read_all(root, names, label, steps):
    result = {}
    for i in range(0,len(names),30):
        args = [UPDATER,'--project','ti60f225_oob.xml','--mode','read']
        for name in names[i:i+30]: args += ['--memory',name]
        text, step = execute(args,root,f'{label}-{i//30}.log');steps.append(step)
        part = decode_read(text)
        assert set(part) == set(names[i:i+30])
        result.update(part)
    return result


def generate(root, label, steps):
    script = root/'generate-overlay.cmd'
    if not script.exists():
        script.write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\ncall efx_run.bat ti60f225_oob.xml --flow pgm --timeout 240\nexit /b %errorlevel%\n')
    _,step = execute(['cmd.exe','/d','/c',script],root,label+'.log',270);steps.append(step)
    bit = root/'outflow/ti60f225_oob.bit'
    assert bit.stat().st_size > 0
    return sha(bit)


def image(path):
    """Efinity hex-encoded byte image: 256-byte textual header then payload.

    Keep all configuration/header settings; ignore only Generated and Project
    text in a *comparison*. Never rewrite the actual released image.
    """
    raw=bytes.fromhex(Path(path).read_text())
    header=raw[:256].decode('ascii')
    fields=[line for line in header.splitlines() if line]
    assert fields[0]=='Version: 2026.1.132'
    assert sum(x.startswith('Generated: ') for x in fields)==1
    assert sum(x.startswith('Project: ') for x in fields)==1
    fixed=[x for x in fields if not x.startswith(('Generated: ','Project: '))]
    assert 'Device: Ti60F225' in fixed and 'PADDED_BITS: 0' in fixed
    return fixed,raw[256:]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--candidate',type=Path,default=CANDIDATE);a=p.parse_args()
    root=a.root.resolve(); assert root.is_relative_to(ROOT/'tmp/frame-execution') and not root.exists()
    candidate=a.candidate.resolve()
    cr=json.loads((candidate/'result.json').read_text(encoding='utf8'))
    for name,h in cr['evidence'].items(): assert sha(name)==h
    parent_bit=sha(BASE/'outflow/ti60f225_oob.bit')
    assert parent_bit=='b9da6275178e6bf83c76ee4fbcdbb2f956aae9986b37c12a0402cb0ce535311b'
    # Some copies use the peripheral XML at project root; include every actual
    # input, but never invent an absent artifact.
    protected=[n for n in GEOMETRY if (BASE/n).exists()]
    protected += [str(f.relative_to(BASE)) for f in BASE.glob('*.peri.xml')]
    inputs={n:sha(BASE/n) for n in protected}
    shutil.copytree(BASE,root)
    for name in ('result.json','generation_inputs.json'):
        shutil.move(root/name,root/('parent_'+name))
    lbf=root/'work_pnr/ti60f225_oob.lbf'; original_lbf=sha(lbf)
    shutil.copy2(lbf,root/'original.lbf')
    mapped=(root/'outflow/ti60f225_oob.map.v').read_text(encoding='utf8')
    names=re.findall(r'^\s*EFX_RAM\w+\s+([^\s]+)\s*\(',mapped,re.M)
    names=[n.lstrip('\\') for n in names]; assert len(names)==len(set(names))==147
    original,blocks=read_rom(mapped)
    native=Path(next(n for n in cr['evidence'] if n.endswith('native_source.v')))
    nw,nb=read_rom(native.read_text(encoding='utf8'));assert nw==original
    low=next(x[4] for x in blocks if x[3]==20)
    steps=[]
    before=read_all(root,names,'before',steps)
    assert before[TARGET]==f'{low:010240b}'[::-1][:len(before[TARGET])]
    assert len(before[TARGET])%20==0 and len(before[TARGET])>=164*20
    assert low>>(len(before[TARGET]))==0
    depth=len(before[TARGET])//20
    values=[(low>>(20*i))&0xfffff for i in range(depth)]
    expected=list(original)
    for address,c in cr['changes'].items():
        i=int(address);assert original[i] == c['register']<<9|c['old']<<1
        expected[i]=c['register']<<9|c['new']<<1
        values[i]=expected[i]&0xfffff
    init=root/'sensor-gain-low20.hex'
    init.write_text(''.join(f'{v:05x}\n' for v in values))
    # The official updater reads/updates a logical memory using its own
    # placement metadata. Do not hand-patch a compressed bitstream or LBF.
    _,step=execute([UPDATER,'--project','ti60f225_oob.xml','--memory',TARGET+','+str(init),
                    '--output',lbf],root,'update.log');steps.append(step)
    after=read_all(root,names,'after',steps)
    assert [n for n in names if before[n]!=after[n]]==[TARGET]
    assert after[TARGET]==''.join(f'{v:020b}'[::-1] for v in values)
    high=after['camera_checked/config_rom/ROM__D$1']
    actual=[int(high[i*5:(i+1)*5][::-1],2)<<20 | int(after[TARGET][i*20:(i+1)*20][::-1],2) for i in range(164)]
    assert actual==expected
    changed_bits=[i for i,(x,y) in enumerate(zip(before[TARGET],after[TARGET])) if x!=y]
    assert changed_bits and {i//20 for i in changed_bits}==set(cr['changed_addresses'])
    updated_lbf=sha(lbf);assert updated_lbf!=original_lbf
    shutil.copy2(lbf,root/'updated.lbf')
    updated_bit=generate(root,'generate-updated',steps)
    assert updated_bit!=parent_bit
    shutil.copy2(root/'outflow/ti60f225_oob.bit',root/'updated.bit')
    # End-to-end reversible control: restoring the untouched routed LBF must
    # recreate the exact original released bitstream with the same pgm flow.
    shutil.copy2(root/'original.lbf',lbf)
    roundtrip=generate(root,'generate-original-control',steps)
    assert image(root/'outflow/ti60f225_oob.bit')==image(BASE/'outflow/ti60f225_oob.bit'), 'pgm payload/settings roundtrip'
    shutil.copy2(root/'updated.lbf',lbf)
    final_bit=generate(root,'generate-final',steps)
    assert image(root/'outflow/ti60f225_oob.bit')==image(root/'updated.bit'), 'repeat pgm payload/settings'
    updated_bit=final_bit
    for name,h in inputs.items(): assert sha(BASE/name)==sha(root/name)==h
    assert sha(BASE/'work_pnr/ti60f225_oob.lbf')==original_lbf
    assert sha(BASE/'outflow/ti60f225_oob.bit')==parent_bit
    result=dict(status='OFFICIAL_BRAM_GAIN_OVERLAY_PASS_NOT_RELEASED',hardware_access=False,board_release=False,
        base=str(BASE),candidate=str(candidate),candidate_sha256=sha(candidate/'result.json'),
        candidate_native_sha256=cr['candidate_native_sha256'],base_geometry=inputs,
        mapping_identity='unchanged base map plus official logical-memory overlay; map does not include new INIT',
        all_logical_memories_compared=len(names),unchanged_memories=len(names)-1,
        target=TARGET,target_logical_depth=depth,changed_bits=changed_bits,changed_addresses=cr['changed_addresses'],
        compared_configuration_words=164,configuration_words=actual,
        exposure_before_half_lines=cr.get('exposure_before_half_lines',2989),
        exposure_after_half_lines=cr.get('exposure_after_half_lines',2989),
        gain_before=cr['gain_before'],gain_after=cr.get('gain_after',cr.get('gain_candidate')),original_lbf_sha256=original_lbf,updated_lbf_sha256=updated_lbf,
        bitstream_sha256=updated_bit,bitstream_bytes=(root/'outflow/ti60f225_oob.bit').stat().st_size,
        original_bitstream_roundtrip_sha256=roundtrip,original_payload_exact_roundtrip=True,
        repeat_updated_payload_exact=True,header_comparison_ignored_only=['Generated','Project'],
        header_format_source=str(EF/'pgm/bin/efx_pgm/header_processor.py'),
        header_format_source_sha256=sha(EF/'pgm/bin/efx_pgm/header_processor.py'),
        original_payload_sha256=hashlib.sha256(image(BASE/'outflow/ti60f225_oob.bit')[1]).hexdigest(),
        updated_payload_sha256=hashlib.sha256(image(root/'outflow/ti60f225_oob.bit')[1]).hexdigest(),
        tool_sha256=sha(UPDATER),runner_sha256=sha(__file__),steps=steps,
        init_file_sha256=sha(init),memory_before={n:__import__('hashlib').sha256(v.encode()).hexdigest() for n,v in before.items()},
        memory_after={n:__import__('hashlib').sha256(v.encode()).hexdigest() for n,v in after.items()})
    (root/'result.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({k:result[k] for k in ('status','all_logical_memories_compared','changed_bits','bitstream_sha256')}),flush=True)


if __name__=='__main__':main()
