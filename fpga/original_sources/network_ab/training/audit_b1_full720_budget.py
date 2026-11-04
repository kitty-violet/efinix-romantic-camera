"""Read-only architecture counterfactual, never a deployable B1 contract.

Keep convolution kernels/channels; remove both decimations and both NN enlarges.
Report mathematical lower bounds, not new RTL synthesis/timing or image quality.
"""
import hashlib
import json
import math
from collections import Counter
from functools import lru_cache
from pathlib import Path

from network_ab.model_contract import compile_contract, load_contract
from network_ab.ddr_memory_map import load_memory_map

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'model-training-workspace/runs/b1-full720-feasibility-20261005'
CONTRACT = ROOT / 'network_ab/model_b_contract.json'
EXPECTED = '5750c4668d75077746ec5d4accf8773ab437540cf933c202f153a8797c21dd01'
FREQ = 150_000_000
MAC_PER_CYCLE = 256
DDR_PEAK = 1_600_000_000


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(plan, full720):
    shapes = {'input': [720, 1280, 3]}
    rows = []
    bykind = Counter()
    for node in plan['nodes']:
        name, kind = node['id'], node['kind']
        h, w, ci = shapes[node['inputs'][0]]
        co = node['channels']
        stride = 1 if full720 else node['stride']
        if node['kernel']:
            shape = [(h + stride - 1)//stride, (w + stride - 1)//stride, co]
            weights = co * node['kernel']**2 * (1 if kind == 'dw3x3' else ci)
            assert weights == node['raw_weights']
            macs = shape[0]*shape[1]*weights
        else:
            weights = macs = 0
            scale = 2 if kind == 'nearest_x2' and not full720 else 1
            shape = [h*scale, w*scale, co]
            if kind == 'residual_add':
                assert shape == shapes[node['inputs'][1]]
        if not full720:
            assert shape == node['output_shape_hwc'] and macs == node['macs']
        shapes[name] = shape
        bykind[kind] += macs
        rows.append(dict(node=name, shape_hwc=shape, weights=weights, macs=macs,
                         int8_tensor_bytes=math.prod(shape)))
    assert shapes['rgb_head'] == [720, 1280, 3]
    byid = {n['id']: n for n in plan['nodes']}

    @lru_cache(None)
    def dependencies(name, x):
        if name == 'input':
            return frozenset([x])
        node = byid[name]
        if node['kind'] == 'nearest_x2':
            return dependencies(node['inputs'][0], x if full720 else x//2)
        if node['kernel']:
            center = x*(1 if full720 else node['stride'])
            offsets = (-1, 0, 1) if node['kernel'] == 3 else (0,)
            return frozenset().union(*(dependencies(node['inputs'][0], center+d) for d in offsets))
        return dependencies(node['inputs'][0], x) | dependencies(node['inputs'][1], x)

    # Exact potential source support away from padding, all four sampling phases.
    rf = []
    for x in range(512, 516):
        support = dependencies('rgb_head', x)
        rf.append(dict(output_x=x, source_min=min(support), source_max=max(support),
                       span=max(support)-min(support)+1, count=len(support)))
    macs = sum(bykind.values())
    feature = math.prod(shapes['down2.pw'])
    # Existing encoder->DDR feature0->residual body->DDR feature1->decoder.
    # Each of the two feature maps is written/read at least once. Excludes halos,
    # AXI padding, commands, camera, display, CPU, gesture, refresh and turnaround.
    payload = 4*feature + 2*720*1280*3
    return dict(nodes=rows, macs_per_frame=macs, by_kind_macs=dict(bykind),
                convolution_weights=sum(r['weights'] for r in rows),
                fp32_trainable_parameter_count=plan['trainable_parameter_count'],
                mac_only_lower_bound_ms=macs/(MAC_PER_CYCLE*FREQ)*1000,
                mac_only_fps_upper_bound=MAC_PER_CYCLE*FREQ/macs,
                required_gmac_per_second_at30=macs*30/1e9,
                required_mac_capacity_multiple_at30=macs*30/(MAC_PER_CYCLE*FREQ),
                mac_dsp_lower_bound_at30_same_packing=math.ceil(macs*30/(2*FREQ)),
                bottleneck_int8_bytes=feature, bottleneck_mib=feature/2**20,
                largest_tensor_mib=max(math.prod(s) for s in shapes.values())/2**20,
                current_fusion_minimum_cnn_ddr_bytes_per_frame=payload,
                current_fusion_minimum_cnn_ddr_gbps_at30=payload*30/1e9,
                ddr_only_fps_upper_bound_same_fusion=DDR_PEAK/payload,
                source_support_1d_four_phases=rf)


def main():
    assert sha(CONTRACT) == EXPECTED
    plan = compile_contract(load_contract())
    current, full = evaluate(plan, False), evaluate(plan, True)
    assert current['macs_per_frame'] == 966_758_400
    assert full['macs_per_frame'] == 720*1280*plan['raw_weight_count'] == 9_451_929_600
    assert full['convolution_weights'] == current['convolution_weights'] == 10256
    assert all(r['span'] == 27 for r in full['source_support_1d_four_phases'])
    mmap = load_memory_map()
    physical = ROOT/'tmp/frame-execution/early-collision-physical1/capacity_gate.json'
    audit = ROOT/'tmp/frame-execution/connected-board-early-collision1/resource_audit2.json'
    frame = ROOT/'tmp/frame-execution/early-collision-frame1/result.json'
    capacity, mapped, measured = [json.loads(p.read_text(encoding='utf-8')) for p in (physical, audit, frame)]
    assert capacity['map_sha256'] == mapped['map_sha256']
    assert mapped['physical_dsp'] == 160 and mapped['ram_blocks'] == 228
    sources = [CONTRACT, Path(__file__), ROOT/'network_ab/model_contract.py',
               ROOT/'hw/efinity/config/ddr_memory_map.json', physical, audit, frame,
               ROOT/'hw/efinity/prototypes/frame_execution/fourway_full_vectors.py',
               ROOT/'hw/efinity/prototypes/frame_execution/fourway_frame_program.py']
    result = dict(status='COUNTERFACTUAL_BUDGET_ONLY_NOT_IMPLEMENTED',
                  changes='stride2 -> stride1; both nearest_x2 -> identity; all kernels/channels retained',
                  source_contract_sha256=EXPECTED, frequency_assumed_hz=FREQ,
                  existing_mac_per_cycle=MAC_PER_CYCLE, physical_ddr_peak_bytes_per_second=DDR_PEAK,
                  existing=current, hypothetical_all720=full,
                  mac_ratio=full['macs_per_frame']/current['macs_per_frame'],
                  feature_ratio=full['bottleneck_int8_bytes']/current['bottleneck_int8_bytes'],
                  same_map_resource_snapshot=dict(capacity=capacity, physical_dsp=160, ram_blocks=228,
                      device=mapped['device'], evidence=str(audit)),
                  existing_cnn_behavioral_ddr_cycles=measured['cycles'],
                  existing_cnn_ms_at_assumed150mhz=measured['measured_frame_ms'],
                  feature_slots=[dict(name=name, reserved_bytes=mmap.region(name).size,
                    full720_fits=full['bottleneck_int8_bytes'] <= mmap.region(name).size)
                    for name in ('feature0', 'feature1')],
                  package_payload_size_if_coeff_layout_retained=17792,
                  package_transport_size_if_coeff_layout_retained=20480,
                  same_package_does_not_make_changed_graph_compatible=True,
                  training_added=0, hardware_changed=False, full720_gpu_quality_test=False,
                  full720_rtl_test=False, full720_pnr_or_sta=False,
                  evidence=[dict(path=str(p), sha256=sha(p)) for p in sources])
    OUT.mkdir(exist_ok=False)
    (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    assert sha(CONTRACT) == EXPECTED
    print(json.dumps({k:result[k] for k in ('status','mac_ratio','feature_slots')},ensure_ascii=False))
    for label, data in [('existing',current),('full720',full)]:
        print(label,json.dumps({k:v for k,v in data.items() if k != 'nodes'},ensure_ascii=False))


if __name__ == '__main__':
    main()
