"""Bind accepted coefficients to existing row-pair descriptors and known answers.

Only the three run-time residual fields change. No RTL, placement, dependencies,
addresses, network topology or accepted package is rewritten.
"""
import argparse
import collections
import hashlib
import json
import shutil
import struct
import time
import zlib
from pathlib import Path
from accepted_a_assets import (ROOT, HERE, ACCEPTED, PACKAGE_SHA, accepted_package,
                               read_json, write_json, write_coefficients, sha)


def descriptors(root, residual):
    source = ROOT/'tmp/frame-execution/paired-ddr-descriptors1'
    proof = read_json(source/'result.json')
    assert sha(source/'result.json') == read_json(ROOT/'tmp/frame-execution/skip-credit-frame1/result.json')['descriptor_manifest_sha256']
    for name, value in proof['files'].items():
        assert sha(source/name) == value
    assert sha(source/'schema.json') == proof['schema_sha256']
    schema = read_json(source/'schema.json')
    program = read_json(Path(proof['program_path'])/'program.json')
    assert sha(Path(proof['program_path'])/'program.json') == proof['program_sha256']
    fields = {f['field']: f for f in schema['compute']['fields']}
    def get(word, name):
        f = fields[name]
        return (word >> f['offset']) & ((1 << f['width'])-1)
    def put(word, name, value):
        f = fields[name]
        assert 0 <= value < 1 << f['width']
        mask = ((1 << f['width'])-1) << f['offset']
        return (word & ~mask) | (value << f['offset'])
    original = (source/'compute.bin').read_bytes()
    words = [int.from_bytes(original[i:i+64], 'little') for i in range(0, len(original), 64)]
    changed = []; counts = collections.Counter()
    for i, word in enumerate(words):
        command = program[get(word, 'program_index')]
        assert command['kind'] == 'compute'
        if not command['skip']:
            assert not (get(word, 'f6') & 16)
            continue
        name = command['node']; r = residual[name]; counts[name] += 1
        flags = (get(word, 'f6') & ~32) | (int(r['main_scaled']) << 5)
        new = put(put(put(word, 'f6', flags), 'f24', r['multiplier']), 'f25', r['post_multiplier'])
        allowed = (1 << (fields['f6']['offset']+5))
        for namefield in ('f24', 'f25'):
            f = fields[namefield]; allowed |= ((1 << f['width'])-1) << f['offset']
        assert ((new ^ word) & ~allowed) == 0
        assert get(new, 'f24') == r['multiplier'] and get(new, 'f25') == r['post_multiplier']
        changed.append(dict(record=i, command=command['command'], node=name,
                            old=[bool(get(word, 'f6') & 32), get(word, 'f24'), get(word, 'f25')], new=r))
        words[i] = new
    assert set(counts) == set(residual)
    root.mkdir()
    raw = b''.join(w.to_bytes(64, 'little') for w in words)
    (root/'compute.bin').write_bytes(raw)
    shutil.copy2(source/'dma.bin', root/'dma.bin')
    assert sha(root/'dma.bin') == proof['files']['dma.bin']
    schema['compute']['crc32'] = f'{zlib.crc32(raw):08x}'
    write_json(root/'schema.json', schema)
    write_json(root/'residual-program-audit.json', dict(counts=dict(counts), records=changed,
              only_residual_runtime_bits_changed=True, unchanged_dependency_and_address_fields=True,
              parameter_source_sha256=PACKAGE_SHA, source_manifest_sha256=sha(source/'result.json')))
    result = dict(proof, status='ACCEPTED_A_RESIDUAL_PROGRAM_BOUND',
                  files={n:sha(root/n) for n in ('compute.bin', 'dma.bin')},
                  schema_sha256=sha(root/'schema.json'), package_sha256=PACKAGE_SHA,
                  crc32={n:f'{zlib.crc32((root/(n+".bin")).read_bytes()):08x}' for n in ('compute','dma')},
                  residual_encoding=residual, residual_records=dict(counts),
                  rtl_decoder_verified=False, full_frame_numeric=False)
    write_json(root/'result.json', result)
    return result


def cluster(root, raw, params, residual):
    """Keep the exact frozen short-fixture addresses/stalls; replace arithmetic."""
    from network_ab.coefficient_layout import build_layout
    from network_ab.contract_reference import requantize
    from validate_dw_block import byte
    base = ROOT/'tmp/frame-execution/skip-credit-cluster1/vectors'
    shutil.copytree(base, root)
    write_coefficients(root, raw)
    operators = {o['weights']['offset']: o for o in build_layout()['operators']}
    rows = (root/'cases.txt').read_text().splitlines()[1:]
    arrays = {n:[int(s,16) for s in (root/(n+'.hex')).read_text().splitlines()]
              for n in ('case_stride','case_stem','case_view','case_fuse','case_main_scaled','case_align',
                        'case_post','case_skip_planes','case_skip_bases','case_skip_delta',
                        'case_bias','case_multiplier','case_relu','golden')}
    checks = 0
    for text in rows:
        v = [int(s, 10 if i < 13 else 16) for i, s in enumerate(text.split())]
        ci, off, pw, w, h, gi, go, mask, sx, origin, sw, span, weight_off, planesword, basesword, _, _ = v
        op = operators[weight_off]; p = params[op['name']]
        planes = [(planesword >> (2*y)) & 3 for y in range(6)]
        bases = [(basesword >> (10*y)) & 1023 for y in range(6)]
        stride = arrays['case_stride'][ci]; stem = arrays['case_stem'][ci]; nn = arrays['case_view'][ci]
        arrays['case_bias'][ci] = sum((b & 0xffffffff) << (32*k) for k,b in enumerate(p.bias))
        arrays['case_multiplier'][ci] = sum(m << (32*k) for k,m in enumerate(p.multiplier))
        arrays['case_relu'][ci] = (1 << len(p.bias))-1 if p.relu else 0
        if arrays['case_fuse'][ci]:
            residual_name = op['name'].split('_')[0]+'_add'; rec = residual[residual_name]
            arrays['case_main_scaled'][ci] = int(rec['main_scaled'])
            arrays['case_align'][ci] = rec['multiplier']; arrays['case_post'][ci] = rec['post_multiplier']
        for y in range(h):
            for x in range(w):
                for g in range(go):
                    values = []
                    for lane in range(8):
                        oc = 8*g+lane
                        if oc >= op['output_channels']:
                            values.append(0); continue
                        dot = 0
                        if pw:
                            for ch in range(op['input_channels']):
                                dot += byte(planes[y]*4+x%4,bases[y]+x//4*gi+ch//8,ch%8,ci)*p.weights[oc][ch]
                        else:
                            for ky in range(3):
                                for kx in range(3):
                                    ry = (2*y if stride else y)+ky
                                    lx = (max(0,min(sw-1,sx+(2*x if stride else x)+kx-1)) >> nn)-origin
                                    b = planes[ry]*4+lx%4; a = bases[ry]+lx//4*gi+g
                                    if stem:
                                        for ic in range(3):
                                            dot += byte(b,a,ic,ci)*p.weights[oc][(3*ky+kx)*3+ic]
                                    else:
                                        dot += byte(b,a,lane,ci)*p.weights[oc][3*ky+kx]
                        value, overflow = requantize(dot,p.bias[oc],p.multiplier[oc],relu=p.relu)
                        assert not overflow
                        if arrays['case_fuse'][ci]:
                            delta=arrays['case_skip_delta'][ci]
                            plane=(arrays['case_skip_planes'][ci]>>(2*y))&3
                            base=(arrays['case_skip_bases'][ci]>>(10*y))&1023
                            skip=byte(plane*4+(x+delta)%4,base+(x+delta)//4*go+g,lane,ci)
                            small, other = (value,skip) if rec['main_scaled'] else (skip,value)
                            aligned=requantize(small,0,rec['multiplier'])[0]
                            value=requantize(aligned+other,0,rec['post_multiplier'])[0]
                        values.append(value); checks += 1
                    arrays['golden'][off+(y*w+x)*go+g] = sum((z & 255) << (8*k) for k,z in enumerate(values))
    for name in ('case_bias','case_multiplier','case_relu','case_main_scaled','case_align','case_post','golden'):
        (root/(name+'.hex')).write_text(''.join(f'{n:x}\n' for n in arrays[name]),encoding='ascii')
    write_json(root/'accepted_cluster.json',dict(package_sha256=PACKAGE_SHA,scalar_reference_checks=checks,
              cases=len(rows),boundary_and_backpressure_fixture='unchanged skip-credit-cluster1',residual_encoding=residual))


def make(root, device):
    import numpy as np
    import torch
    from PIL import Image
    from network_ab.training.b1_trained_image import runtime_from_image
    from network_ab.training.probe_b1_cartoon_int8 import IntegerB1
    raw, params, residual, plan = accepted_package()
    assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    root.mkdir(parents=True)
    before={p.relative_to(ACCEPTED).as_posix():sha(p) for p in ACCEPTED.rglob('*') if p.is_file()}
    write_json(root/'accepted-inputs.json',before)
    d = descriptors(root/'descriptors',residual)
    cluster(root/'cluster-vectors',raw,params,residual)
    base = ROOT/'tmp/frame-execution/joint-reorder-vectors1'
    vm = read_json(base/'frame_manifest.json')
    commands = read_json(base/'program.json')
    packed=[int(s,16) for s in (base/'program.hex').read_text().splitlines()]
    assert len(commands)==len(packed)==19229
    for i,c in enumerate(commands):
        if c['kind']!='compute' or not c['skip']:continue
        r=residual[c['node']];v=packed[i];v &= ~(1 << (6*32+5));v |= int(r['main_scaled']) << (6*32+5)
        for k,value in ((24,r['multiplier']),(25,r['post_multiplier'])):
            v=(v & ~(0xffffffff << (k*32))) | value << (k*32)
        packed[i]=v
    serial=''.join(f'{v:0256x}\n' for v in packed)
    torch.set_num_threads(2)
    engine=IntegerB1(runtime_from_image(raw),device=device)
    results=[]
    with torch.inference_mode():
        for item in read_json(ACCEPTED/'validation/data.json'):
            out=root/item['id'];out.mkdir();write_coefficients(out,raw)
            source=np.asarray(Image.open(ACCEPTED/item['source']).convert('RGB'))
            expected=np.asarray(Image.open(ACCEPTED/item['output']).convert('RGB'))
            assert source.shape==expected.shape==(720,1280,3)
            q=(source.astype(np.int16)-128).astype(np.int8);q.tofile(out/'input.bin')
            tensors,_=engine(torch.tensor(q.transpose(2,0,1)[None].copy(),device=device,dtype=torch.float32),True)
            metadata=[];offset=0
            with (out/'golden.bin').open('wb') as f:
                for n in plan['nodes']:
                    z=tensors[n['id']][0].permute(1,2,0).cpu().numpy()
                    assert np.equal(z,z.astype(np.int8)).all();z=z.astype(np.int8)
                    assert list(z.shape)==n['output_shape_hwc']
                    pad=np.pad(z,((0,0),(0,0),(0,(-z.shape[2])%8)))
                    old=next(v for v in vm['nodes'] if v['id']==n['id']);assert old['offset']==offset
                    metadata.append(dict(old,sha256=hashlib.sha256(z.tobytes()).hexdigest(),unique_values=int(np.unique(z).size)))
                    f.write(pad.tobytes());offset+=pad.size
                final=(z.astype(np.int16)+128).astype(np.uint8)
                error=int(np.abs(final.astype(np.int16)-expected.astype(np.int16)).max())
                assert error==0,(item['id'],'accepted output mismatch',error)
            assert offset==81100800
            (out/'program.hex').write_text(serial,encoding='ascii')
            for name in ('compute_commands.hex','compute_dependencies.hex','dma_commands.hex','dma_dependencies.hex'):
                assert sha(base/name)==vm['files'][name];shutil.copy2(base/name,out/name)
            manifest=dict(vm,accepted_style_revision='B1-A-20261008',trained_weights=False,
                          weights='user-accepted constructed B1 coefficients, calibrated RGB head',
                          package_sha256=PACKAGE_SHA,nodes=metadata,residual_encoding=residual,
                          residual=list(residual.values()),source_sha256=sha(ACCEPTED/item['source']),
                          accepted_output_sha256=sha(ACCEPTED/item['output']),accepted_image_error=error,
                          descriptor_manifest_sha256=sha(root/'descriptors/result.json'))
            manifest['files']={p.name:sha(p) for p in out.iterdir() if p.is_file()}
            write_json(out/'frame_manifest.json',manifest)
            results.append(dict(id=item['id'],image_error=error,nodes=len(metadata),manifest_sha256=sha(out/'frame_manifest.json')))
            print('ACCEPTED_VECTOR_READY',item['id'],flush=True)
            del tensors
    assert before=={p.relative_to(ACCEPTED).as_posix():sha(p) for p in ACCEPTED.rglob('*') if p.is_file()}
    write_json(root/'result.json',dict(status='ACCEPTED_A_PACKAGE_PROGRAM_AND_KNOWN_ANSWERS_READY',
              package_sha256=PACKAGE_SHA,padded_crc32=f'{zlib.crc32(raw):08x}',
              descriptor_crc32=d['crc32'],residual_records=d['residual_records'],cases=results,
              accepted_unchanged=True,rtl_pass=False,actual_board=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');a=p.parse_args()
    make(a.root.resolve(),a.device)
