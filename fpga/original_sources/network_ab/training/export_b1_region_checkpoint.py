"""Export one explicit checkpoint, with source provenance and byte replay.

This does not select a winner, infer a cumulative training budget, or release
hardware. The existing exporter verifies every node and an independent oracle.
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import torch

from . import train_b1_minimum_area as m
from .publish_b1_style_recovery import export
from .runtime import configure_workspace, contract_guard, file_hash, write_json


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    ap.add_argument('--data', type=Path, help='Review manifest; defaults to the checkpoint training manifest')
    args = ap.parse_args()
    configure_workspace(); contract_guard()
    torch.set_num_threads(2); cv2.setNumThreads(2)
    assert not args.out.exists(), 'Use an independent output directory'
    digest = file_hash(args.checkpoint)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    assert state.get('residual_encoding'), 'Do not silently use default residual arithmetic'
    net = m.t.r.c.ColourB1(revive=False, source=m.t.r.OUT/'initial_identity.bin').to(args.device)
    net.load_state_dict(state['model'])
    net.package['residual_encoding'] = state['residual_encoding']
    net.quantized = True
    review_data = args.data or Path(state.get('args', {}).get('data_manifest') or m.OUT/'data.json')
    data_digest = file_hash(review_data)
    review_rows = m.load_data(review_data)['validation']
    export(net, review_rows, args.out)
    assert file_hash(review_data) == data_digest
    raw = (args.out/'coefficients-padded.bin').read_bytes()
    parameters, residual = m.t.r.decode(raw)
    assert len(raw) == 20480 and len(parameters) == 24
    assert not any(raw[17792:])
    (args.out/'coefficients.bin').write_bytes(raw[:17792])
    assert file_hash(args.checkpoint) == digest
    shutil.copy2(args.checkpoint, args.out/'source-checkpoint.pt')
    assert file_hash(args.out/'source-checkpoint.pt') == digest
    config = args.checkpoint.parent/'config.json'
    source_config = json.loads(config.read_text(encoding='utf-8')) if config.exists() else None
    write_json(args.out/'checkpoint.json', dict(
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=digest,
        step_in_source_run=state.get('step'), source_args=state.get('args'),
        review_data_manifest=str(review_data.resolve()),review_data_sha256=data_digest,
        review_targets=[dict(id=row['id'],path=row['target'],sha256=row['target_sha256']) for row in review_rows],
        source_config=source_config, source_config_sha256=file_hash(config) if config.exists() else None,
        exported_from_fp32_parameters=True, integer_inference=True,
        source_quantization_aware=bool(state.get('qat') or state.get('args', {}).get('qat')),
        manual_coefficient_changes={k:state[k] for k in ('manual_decoder_change','manual_boundary_phase_span',
            'boundary_field_kernel','parent_checkpoint','parent_sha256','added_training_steps',
            'ternary_spatial_vote','final_coordinate_range','manually_calibrated_final_precision',
            'colour_lattice_fit','encoder_antialias_keys','trainable_ordinal_precision',
            'binary_coverage720','ordered_distance_gain','decoder_ramp_gain',
            'coincident_boundary_supervision','coincident_loss_weight',
            'rgb_profile_target_residual_preserved') if k in state},
        residual_encoding=residual, effective_bytes=17792, transport_bytes=len(raw),
        training_gradient_method={k:state[k] for k in ('rgb_priority_gradient',
            'auxiliary_gradient_norm_limit','gradient_projection_scope',
            'raw_rgb_gradient_preserved','adam_monotonicity_guaranteed') if k in state},
        network_representation={k:state[k] for k in (
            'direct_luminance_planes','nonnegative_luma_decoder','hard_luminance_decisions',
            'independent_chroma_decisions','radix_luma_continuous_chroma',
            'hard_chroma_decisions_removed','luminance_layers','chroma_target',
            'trainable_luma_readout','luma_supervision_node','encoder32_channels_are_latent',
            'backend_palette_node','fixed_readout_identity_removed',
            'revived_reserved_luma_features','duplicated_source_channel',
            'source_channel_chosen_on_train_ids','revival_readout_weights',
            'initial_rgb_exactly_preserved','final_rendered_luma_supervision',
            'rendered_luma_reference','rendered_luma_node','rendered_luma_weights',
            'native_region_luma_labels','luminance_label','crop_independent_luma_labels',
            'flow_transports_luma_codes_directly','original_rgb_target_unchanged',
            'shared_luma_alignment360','shared_luma_alignment_weight',
            'shared_luma_alignment_node','shared_luma_alignment_target',
            'fine_full_decoder_trainable','live_luma_precision_refined','final_rgb_training',
            'fixed_palette','latent_geometry_supervision','local_fit_only','geometry_profile',
            'source_phase_auxiliary_heads','auxiliary_heads_exported','contour_coherence_weight',
            'synthetic_data_inference_operations') if k in state},
        cumulative_training_steps_claimed=False, visual_acceptance=False,
        actual_board=False, extra_inference_operations=0))
    (args.out/'README.txt').write_text(
        'B1 实际系数研究候选\n\n'
        'coefficients.bin：17792 字节有效系数。\n'
        'coefficients-padded.bin：20480 字节传输镜像。\n'
        'load_words.hex：2560 个小端 64 位装载字。\n'
        'checkpoint.json 记录源检查点、训练配置和哈希，不推测累计训练步数。\n'
        'review.json 记录六张整图逐节点复演及独立数值参考检查。\n'
        '没有新增推理滤镜。数值一致不等于画质、RTL、时序或实板验收通过。\n', encoding='utf-8')
    print('B1_CHECKPOINT_EXPORT_PASS', file_hash(args.out/'coefficients-padded.bin'), flush=True)


if __name__ == '__main__':
    main()
