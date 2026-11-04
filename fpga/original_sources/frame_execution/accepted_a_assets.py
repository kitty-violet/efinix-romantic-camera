"""Read-only acceptance/ABI boundary for the user-selected B1 style revision A."""
import hashlib
import json
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'network_ab'))
from network_ab.coefficient_layout import build_layout
from network_ab.model_contract import compile_contract, load_contract
from network_ab.training.b1_trained_image import decode

ACCEPTED = ROOT / 'model-training-workspace/accepted/B1-A-20261008'
PACKAGE_SHA = 'ceebe7b3187fd47fbf0924883d19a9a335923d9de378ae09d30f362d9b9d0523'
CONTRACT_SHA = '5750c4668d75077746ec5d4accf8773ab437540cf933c202f153a8797c21dd01'
CANONICAL_SHA = 'b330b3bce8d09544d596ff446a457c60c77407ded12800cd9cceb16fba92abf3'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def accepted_package():
    acceptance = read_json(ACCEPTED/'acceptance.json')
    raw = (ACCEPTED/'coefficients-padded.bin').read_bytes()
    assert acceptance['accepted_by'] == 'user' and acceptance['scope'] == 'current_visual_result'
    assert acceptance['package_sha256'] == PACKAGE_SHA == hashlib.sha256(raw).hexdigest()
    assert len(raw) == 20480 and raw[:17792] == (ACCEPTED/'coefficients.bin').read_bytes()
    assert not any(raw[17792:])
    lines = (ACCEPTED/'load_words.hex').read_text('ascii').splitlines()
    assert len(lines) == 2560 and all(len(s) == 16 for s in lines)
    assert b''.join(int(s, 16).to_bytes(8, 'little') for s in lines) == raw
    assert sha(ROOT/'network_ab/model_b_contract.json') == CONTRACT_SHA == acceptance['contract_file_sha256']
    plan = compile_contract(load_contract())
    assert plan['contract_sha256'] == CANONICAL_SHA == acceptance['canonical_contract_sha256']
    assert plan == read_json(ACCEPTED/'contract/canonical_graph.json')
    assert build_layout() == read_json(ACCEPTED/'contract/coefficient_layout.json')
    params, residual = decode(raw)
    assert residual == read_json(ACCEPTED/'residual_encoding.json')
    assert len(params) == 24 and len(plan['nodes']) == 29
    assert all(r == dict(main_scaled=False, multiplier=1, post_multiplier=131071) for r in residual.values())
    return raw, params, residual, plan


def write_coefficients(folder, raw):
    folder.mkdir(parents=True, exist_ok=True)
    params, residual = decode(raw)
    assert hashlib.sha256(raw).hexdigest() == PACKAGE_SHA
    (folder/'coefficients.bin').write_bytes(raw[:17792])
    (folder/'coefficients-padded.bin').write_bytes(raw)
    words = [int.from_bytes(raw[i:i+8], 'little') for i in range(0, len(raw), 8)]
    (folder/'load_words.hex').write_text(''.join(f'{w:016x}\n' for w in words), encoding='ascii')
    for bank in range(5):
        values = words[bank::5]
        assert len(values) == 512
        (folder/f'bank{bank}.hex').write_text(''.join(f'{w:016x}\n' for w in values), encoding='ascii')
    layout = dict(build_layout(), sha256=hashlib.sha256(raw[:17792]).hexdigest(),
                  package_sha256=PACKAGE_SHA, accepted_style_revision='B1-A-20261008',
                  padded_crc32=f'{zlib.crc32(raw):08x}', residual_encoding=residual)
    write_json(folder/'coefficient_layout.json', layout)
    return layout


def verify_readback(folder, readback):
    raw, _, _, _ = accepted_package()
    assert (folder/'coefficients-padded.bin').read_bytes() == raw
    assert (folder/'coefficients.bin').read_bytes() == raw[:17792]
    banks = [[int(s, 16) for s in (folder/f'bank{b}.hex').read_text().splitlines()] for b in range(5)]
    assert all(len(b) == 512 for b in banks)
    words = [int.from_bytes(raw[i:i+8], 'little') for i in range(0, len(raw), 8)]
    assert all(words[i] == banks[i % 5][i // 5] for i in range(2560))
    lines = Path(readback).read_text('ascii').splitlines()
    assert len(lines) == 2560
    actual = b''.join(int(s, 16).to_bytes(8, 'little') for s in lines)
    assert actual == raw, 'Actual resident RAM readback differs from accepted package'
    decode(actual)
    print(f'ACTUAL_RTL_READBACK_SOFTWARE_VERIFIED accepted=B1-A-20261008 words=2560 '
          f'crc32={zlib.crc32(raw):08x} sha256={PACKAGE_SHA}', flush=True)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--image-dir', required=True, type=Path)
    p.add_argument('--readback', required=True, type=Path)
    a = p.parse_args()
    verify_readback(a.image_dir, a.readback)
