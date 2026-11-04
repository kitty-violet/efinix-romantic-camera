"""Read-only validation of a delivered real B1 package in this project checkout."""
import argparse
import hashlib
import json
from pathlib import Path

from .b1_trained_image import decode, encode


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(package):
    package = package.resolve(strict=True)
    manifest = json.loads((package / 'delivery_manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        path = (package / name).resolve(strict=True)
        if not path.is_relative_to(package):
            raise ValueError('manifest path leaves package')
        if path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']:
            raise ValueError(f'delivery file mismatch: {name}')
    image = (package / 'coefficients-padded.bin').read_bytes()
    if len(image) != 20480:
        raise ValueError('transport must be 20 KiB')
    parameters, residual = decode(image)
    raw = encode(parameters, residual)
    if raw != (package / 'coefficients.bin').read_bytes() or raw + bytes(2688) != image:
        raise ValueError('canonical image mismatch')
    words = (package / 'load_words.hex').read_text().splitlines()
    if len(words) != 2560 or b''.join(int(w, 16).to_bytes(8, 'little') for w in words) != image:
        raise ValueError('transport word order mismatch')
    verification = json.loads((package / 'verification.json').read_text())
    rtl = json.loads((package / 'rtl-package-check/result.json').read_text())
    execution = json.loads((package / 'execution/manifest.json').read_text())
    digest = sha(package / 'coefficients-padded.bin')
    if not (verification['sha256'] == rtl['weights_sha256'] == execution['coefficient_sha256'] == digest):
        raise ValueError('evidence belongs to different weights')
    contract = Path(__file__).parents[1] / 'model_b_contract.json'
    if sha(contract) != verification['contract_sha256']:
        raise ValueError('current network contract changed')
    if any(row['max_bytes_vs_qat_forward'] != 0 for row in verification['checks']):
        raise ValueError('QAT and delivered bytes differ')
    for name, record in residual.items():
        if record != execution['scales'][name]:
            raise ValueError('execution residual scales mismatch')
    for name, expected in execution['files'].items():
        if sha(package / 'execution' / name) != expected['sha256']:
            raise ValueError('execution file mismatch')
    return dict(status='PASS', transport_bytes=len(image), layers=len(parameters),
                residual_records=len(residual), weight_sha256=digest,
                checked_files=len(manifest['files']), actual_board_tested=False,
                visual_quality_accepted=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.package), ensure_ascii=False, indent=2))
