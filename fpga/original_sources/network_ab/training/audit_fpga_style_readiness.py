"""Read-only deployment audit; outputs belong to model-training, never hardware.

Random B1 execution below verifies structure/folding only, not image quality.
Residual arithmetic tests use explicitly labelled diagnostic fixtures.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HW = ROOT / 'hw/efinity/prototypes/frame_execution'
sys.path[:0] = [str(ROOT), str(ROOT / 'network_ab'), str(HW)]
from network_ab.model_contract import compile_contract, load_contract
from network_ab.contract_torch import build_fp32_model
from coefficient_layout import build_layout, decode_image, LayoutError
from residual_q17_align import encode_alignment, fixtures
from contract_reference import residual_value


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def receptive_support(plan):
    # Exact interval propagation through nearest repetition, including phase.
    length = 2048
    values = {'input': np.stack([np.arange(length)] * 2, axis=1)}
    for row in plan['nodes']:
        a = values[row['inputs'][0]]
        if row['kind'] == 'nearest_x2':
            out = np.repeat(a, 2, axis=0)
        elif row['kind'] == 'residual_add':
            b = values[row['inputs'][1]]
            out = np.stack([np.minimum(a[:, 0], b[:, 0]), np.maximum(a[:, 1], b[:, 1])], axis=1)
        else:
            kernel, stride = row['kernel'], row['stride']
            ix = np.arange((len(a) + stride - 1) // stride) * stride
            out = np.stack([a[np.clip(ix-kernel//2, 0, len(a)-1), 0],
                            a[np.clip(ix+kernel//2, 0, len(a)-1), 1]], axis=1)
        values[row['id']] = out
    output = values[plan['nodes'][-1]['id']]
    return {'method': 'exact graph support with eval/folded BN; interior output phases modulo 4',
            'phases': [{'output_index': i, 'source_interval': output[i].tolist(),
                        'support_width': int(output[i, 1]-output[i, 0]+1)} for i in range(1024, 1028)]}


def residual_checks():
    results = []
    a = np.arange(-128, 128, dtype=np.int64)[:, None]
    b = np.arange(-128, 128, dtype=np.int64)[None, :]
    def rq(v, m):
        product = v*m
        return np.clip(np.sign(product)*((np.abs(product)+65536)//131072), -128, 127)
    for fixture in fixtures():
        record = encode_alignment(fixture)
        aligned = rq(a if record['main_scaled'] else b, record['multiplier'])
        actual = rq(aligned+(b if record['main_scaled'] else a), record['post_multiplier'])
        ratio = min(Fraction(*fixture['main_scale']), Fraction(*fixture['skip_scale'])) / Fraction(*fixture['common_scale'])
        small = a if record['main_scaled'] else b
        exact = np.sign(small)*((2*np.abs(small)*ratio.numerator+ratio.denominator)//(2*ratio.denominator))
        reference = rq(exact+(b if record['main_scaled'] else a), record['post_multiplier'])
        assert np.array_equal(actual, reference)
        for x, y in [(-128, -128), (-128, 127), (127, 127), (1, -1), (3, 5)]:
            assert actual[x+128, y+128] == residual_value(x, y, fixture)
        results.append({**record, 'signed_pairs_checked': 65536, 'max_difference': 0})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    assert out.is_relative_to(ROOT/'model-training-workspace/runs') and not out.exists()
    out.mkdir()
    contract = load_contract()
    plan = compile_contract(contract)
    layout = build_layout()
    board = ROOT/'tmp/frame-execution/board-production-full1'
    payload = ROOT/'tmp/frame-execution/board-payload2'
    metadata = json.loads((board/'inputs.json').read_text())
    # Bind audited residual/scale implementation to the actual board snapshot.
    selected = {'pw_skip_fuse4.sv', 'compute_column_cluster.sv',
                'requantize_i32_mul18_q17_pipeline.sv', 'rgb_area_downscaler_3to2.sv',
                'rgb_bilinear_scaler_prototype.sv'}
    snapshots = [r for r in metadata['inputs'] if Path(r['snapshot']).name in selected]
    assert {Path(r['snapshot']).name for r in snapshots} == selected
    for row in snapshots:
        assert sha(row['snapshot']) == row['sha256']
    pm = json.loads((payload/'payload.json').read_text())
    assert pm['network_contract_sha256'] == sha(ROOT/'network_ab/model_b_contract.json')
    for row in pm['payloads']:
        assert sha(payload/row['file']) == row['sha256']
    image = (payload/'coefficients-padded.bin').read_bytes()
    assert len(image) == 20480 and not any(image[layout['total_bytes']:])
    decoded = decode_image(image[:layout['total_bytes']])
    try:
        decode_image(image[:layout['total_bytes']], require_complete_network=True)
    except LayoutError as exc:
        rejection = str(exc)
    else:
        raise AssertionError('Diagnostic package was accepted as trained production')
    dm = json.loads((ROOT/'tmp/frame-execution/production-descriptors3/result.json').read_text())
    schema = dm['streams']['compute']['schema']
    assert sha(payload/'compute.bin') == dm['streams']['compute']['sha256']
    raw = (payload/'compute.bin').read_bytes()
    residual_fields = Counter()
    for offset in range(0, len(raw), 64):
        record = int.from_bytes(raw[offset:offset+64], 'little'); fields = {}
        for name, bits in schema:
            fields[name] = record & ((1 << bits)-1); record >>= bits
        assert record == 0
        if fields['f6'] & 16:
            residual_fields[(fields['f1'], bool(fields['f6'] & 32), fields['f24'], fields['f25'])] += 1
    torch.manual_seed(20261003)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = build_fp32_model(contract).cuda().eval()
    # Non-default stats exercise actual eval BN folding rather than identity BN.
    for layer in model.modules():
        if isinstance(layer, torch.nn.BatchNorm2d):
            layer.running_mean.copy_(torch.linspace(-.04, .04, layer.num_features, device='cuda'))
            layer.running_var.copy_(torch.linspace(.6, 1.4, layer.num_features, device='cuda'))
    folded = copy.deepcopy(model)
    for sequence in folded.operators.values():
        for i, layer in enumerate(sequence):
            if isinstance(layer, torch.nn.BatchNorm2d):
                sequence[i-1] = torch.nn.utils.fuse_conv_bn_eval(sequence[i-1], layer)
                sequence[i] = torch.nn.Identity()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        x = torch.rand(1, 3, 720, 1280, device='cuda')*1.9921875-1
        y = model(x); z = folded(x)
        assert y.shape == (1, 3, 720, 1280) and torch.isfinite(y).all()
        max_error = float((y-z).abs().max())
        assert max_error < 1e-5
    teacher = ROOT/'model-training-workspace/runs/texture-recovery-20261003/final/refined_style_fp32.pt'
    teacher_hash = sha(teacher)
    assert teacher_hash == 'c86b9b99e901c0ad6be7ce74bd35388a5028dc8c12228c579489025fa6b2f7a5'
    extra = [Path(__file__), ROOT/'network_ab/model_b_contract.json', ROOT/'network_ab/contract_torch.py',
             ROOT/'network_ab/coefficient_layout.py', HW/'residual_q17_align.py', HW/'residual_tables.py',
             HW/'fourway_full_vectors.py', HW/'prepare_board_payload.py',
             ROOT/'network_ab/training/smooth_surface_guard.py',
             ROOT/'model-training-workspace/runs/highlight-diagnosis-20261003/correction_config.json']
    result = {
        'status': 'READINESS_AUDIT_PASS_NOT_DEPLOYMENT_OR_QUALITY_PASS',
        'model_id': plan['model_id'], 'contract_file_sha256': sha(ROOT/'network_ab/model_b_contract.json'),
        'contract_canonical_sha256': plan['contract_sha256'],
        'graph_nodes': len(plan['nodes']), 'quantized_operators': plan['quantized_operator_count'],
        'raw_weight_count': plan['raw_weight_count'], 'trainable_parameter_count': plan['trainable_parameter_count'],
        'macs_per_720p_frame': plan['macs_per_frame'], 'coefficient_layout': layout,
        'receptive_support': receptive_support(plan),
        'cuda_structural_smoke': {'device': torch.cuda.get_device_name(), 'torch': torch.__version__,
            'shape': list(y.shape), 'dtype': str(y.dtype), 'bn_fold_max_error': max_error,
            'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20,
            'random_parameters_not_trained': True, 'quality_verified': False},
        'residual_diagnostic_checks': residual_checks(),
        'actual_board_residual_descriptor_fields': [
            dict(node_index=k[0], main_scaled=k[1], align_q17=k[2], post_q17=k[3], occurrences=v)
            for k, v in sorted(residual_fields.items())],
        'current_payload_is_diagnostic': pm['diagnostic_weights'], 'complete_network_rejection': rejection,
        'teacher_checkpoint_sha256': teacher_hash, 'teacher_convolution_parameters': 1043648,
        'required_new_work': ['accepted-teacher-plus-correction targets', 'B1 FP32 visual distillation',
            'hardware-constrained folded-BN QAT', 'trained coefficient export with production semantic validation',
            'residual runtime fields in compute descriptors and refreshed CRC/hash', 'integer/RTL pixel agreement'],
        'hardware_snapshot_binding': snapshots,
        'evidence': {str(p): sha(p) for p in extra},
        'no_hardware_changes': True, 'no_training_started': True, 'no_new_dependencies': True}
    (out/'audit.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['status','raw_weight_count','trainable_parameter_count',
        'macs_per_720p_frame','receptive_support','cuda_structural_smoke','actual_board_residual_descriptor_fields']}, indent=2))


if __name__ == '__main__':
    main()
