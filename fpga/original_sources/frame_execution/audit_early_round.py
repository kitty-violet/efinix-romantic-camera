"""Close this frozen candidate's two full-frame profiles and matching local gates.

This audits evidence, not physical fit. A measured performance failure is retained
as a failure field rather than preventing the numerical evidence from being saved.
"""
import collections
import json
import re
from pathlib import Path

from run_units import ROOT, sha
from audit_joint_throughput import bound_run

BASE = ROOT / 'tmp/frame-execution'
OUT = ROOT / 'output/toolchain/production_early_round_verified_2026-10-01.json'
evidence = []


def read(path):
    evidence.append(dict(path=str(path), sha256=sha(path)))
    return json.loads(path.read_text(encoding='utf-8'))


def bind_files(records, names=None, snapshot_key='sha256'):
    matched = set()
    for item in records:
        path = Path(item['snapshot'])
        assert sha(path) == item[snapshot_key], path
        if names and path.name in names and not path.name.startswith('tb_'):
            assert item[snapshot_key] == names[path.name], path.name
            matched.add(path.name)
    return matched


def local(folder, status, names, module, log_key='log_sha256', snapshot_key='sha256'):
    base = BASE / folder
    r, m = read(base / 'result.json'), read(base / 'inputs.json')
    assert r['status'] == status
    assert r['inputs_sha256'] == sha(base / 'inputs.json')
    assert r[log_key] == sha(base / 'simulation.log')
    assert module in bind_files(m['inputs'], names, snapshot_key)
    return r


def mapped(folder, top, names):
    base = BASE / folder
    r, m = read(base / 'result.json'), read(base / 'inputs.json')
    assert r['status'] == 'ISOLATED_PARALLEL_CANDIDATE_MAP_REQUIRES_INTEGRATION'
    assert r['inputs_sha256'] == sha(base / 'inputs.json')
    for suffix, key in [('.map.v', 'map_sha256'), ('.map.rpt', 'report_sha256'), ('.warn.log', 'warning_sha256')]:
        assert sha(base / 'outflow' / (top + suffix)) == r[key]
    assert all(Path(item['snapshot']).name in names for item in m['inputs'])
    assert top + '.sv' in bind_files(m['inputs'], names)
    net = (base / 'outflow' / (top + '.map.v')).read_text()
    actual = dict(collections.Counter(re.findall(r'^\s*EFX_([A-Z0-9_]+)\s+\S+', net, re.M)))
    assert actual == r['resources']
    report = (base / 'outflow' / (top + '.map.rpt')).read_text()
    for kind, count in actual.items():
        assert re.findall(r'^EFX_' + kind + r'\s*:\s*(\d+)\s*$', report, re.M) == [str(count)]
    return actual


def warnings_review(base):
    text = (base / 'compile.log').read_text()
    rows = re.findall(r'^%Warning-([A-Z0-9_]+): (.*?):(\d+):(\d+): (.*)$', text, re.M)
    reviewed = []
    reasons = {
        'efx_dsp24.v': 'Official primitive model: intentional signed/unsigned extension or parameter string zero-extension.',
        'efx_ram10.v': 'Official primitive model: parameter string zero-extension before comparison.',
        'efx_dpram10.v': 'Official primitive model: parameter string zero-extension before comparison.',
        'shared_rq_oc8_cluster.sv': 'Bounded unsigned group 0..3 compared with loop index 0..3.',
        'finish_rr_arbiter.sv': 'Bounded unsigned owner compared with constant NUM_FRONTS-1.',
        'shared_pool_banked_fifo.sv': 'Bounded unsigned owner 0..7 compared with loop index 0..7.',
        'frame_arena_row_dma.sv': 'Unsigned 4KiB boundary beat count extended to 16-bit unsigned burst count.',
        'next_header_prefetch.sv': 'Unsigned bounded header indices extended to loop-index arithmetic; field guard selects valid bytes.',
        'compute_column_cluster.sv': 'Unsigned bounded pixel/batch arithmetic is explicitly sized at final 13-bit assignment.',
        'unified_column_dispatch32.sv': 'Generate owner P=0 or DP=0 specializes an unsigned >=0 comparison to true.',
    }
    for code, path, line, column, message in rows:
        name = Path(path).name
        assert code in ('WIDTHEXPAND', 'UNSIGNED'), (code, name, line)
        assert name in reasons, (name, line)
        assert (code == 'UNSIGNED') == (name == 'unified_column_dispatch32.sv')
        reviewed.append(dict(code=code, file=name, line=int(line), message=message, rationale=reasons[name]))
    manifest = json.loads((base / 'inputs.json').read_text())
    assert dict(collections.Counter(x[0] for x in rows)) == manifest['warning_codes']
    return dict(compile_log_sha256=sha(base / 'compile.log'), count=len(rows), items=reviewed)


def main():
    assert not OUT.exists(), 'Keep completed evidence immutable'
    ideal_base = BASE / 'production-early-frame1'
    stall_base = BASE / 'production-early-stall-frame1'
    r0, m0, vm = bound_run(ideal_base)
    r1, m1, _ = bound_run(stall_base)
    for base in (ideal_base, stall_base):
        for name in ('result.json', 'inputs.json'):
            read(base / name)
    assert r0['profile'] == m0['profile'] == 0 and r1['profile'] == m1['profile'] == 1
    assert m0['binary_sha256'] == m1['binary_sha256']
    assert m0['inputs'] == m1['inputs'] and m0['vector_manifest_sha256'] == m1['vector_manifest_sha256']
    assert r0['line'][1] == r1['line'][1] == '19229'
    assert r0['line'][3:] == r1['line'][3:]
    names = {Path(x['snapshot']).name: x['sha256'] for x in m0['inputs']}
    core = mapped('production-early-map1', 'compute_column_cluster', names)
    dma = mapped('production-dma-map2', 'frame_arena_row_dma', names)
    reader = local('production-early-fourstate1', 'FOUR_STATE_READER_PACKET_EQUIVALENCE_PASS', names,
                   'unified_column_reader.sv', snapshot_key='snapshot_sha256')
    dispatch = local('production-onehot-unit1', 'UNIFIED_DISPATCH_CYCLE_EQUIVALENCE_PASS', names,
                     'unified_column_dispatch32.sv')
    dma_gate = local('production-dma-source3', 'ROW_DMA_OFFICIAL_RAM_SOURCE_PASS', names,
                     'frame_arena_row_dma.sv', log_key='simulation_sha256')
    native_base = BASE / 'production-early-local1'
    native, nm = read(native_base / 'result.json'), read(native_base / 'native_inputs.json')
    assert native['status'] == 'NATIVE_LOCAL_CLUSTER_NUMERIC_PROTOCOL_PASS'
    assert native['inputs_sha256'] == sha(native_base / 'native_inputs.json')
    assert native['log_sha256'] == sha(native_base / 'simulation.log')
    assert sha(nm['binary']) == nm['binary_sha256']
    source_manifest = Path(nm['snapshot']) / 'inputs.json'
    assert sha(source_manifest) == nm['source_manifest_sha256']
    assert 'compute_column_cluster.sv' in bind_files(read(source_manifest)['inputs'], names)
    dep_path = Path(m0['vectors']) / 'dependency_audit.json'
    dep = read(dep_path)
    assert dep['status'] == 'FINITE_DDR_DEPENDENCY_COVERAGE_PASS'
    assert dep['vector_manifest_sha256'] == m0['vector_manifest_sha256']
    capacity = ROOT / 'tmp/efinity-smoke-2026.1/sc431-v23-fifo2026-seed1-effort5/outflow/ti60f225_oob.place.rpt'
    assert re.search(r'^XLRs:\s*\d+ / 60800 ', capacity.read_text(), re.M)
    evidence.append(dict(path=str(capacity), sha256=sha(capacity)))
    parts = {0: 'tail/commit', 1: 'PW', 2: 'DW/stem/stride', 3: 'PW/residual', 5: 'unhidden DMA', 6: 'header/config'}
    measurements = []
    for r in (r0, r1):
        measurements.append(dict(profile=r['profile'], cycles=r['cycles'], ms_at_target_150mhz=r['measured_frame_ms'],
            budget_pass=r['cycles'] <= 5000000, margin_cycles=5000000-r['cycles'],
            margin_ms=(5000000-r['cycles'])/150000,
            exclusive=[dict(name=parts.get(x['id'], 'unused reason '+str(x['id'])), cycles=x['cycles'], ms=x['cycles']/150000) for x in r['reasons']],
            per_node=r['nodes'], stalls=r['stalls'], dma_overlap=r['dma_overlap']))
    result = dict(status='BOUND_EARLY_ROUND_NUMERIC_EVIDENCE_PASS', audit_runner_sha256=sha(__file__), measurements=measurements,
        core_resources=core, dma_resources=dma, device_xlr_capacity=60800,
        core_physical_dsp=core['DSP24']//2+core['DSP48'],
        commands=vm['commands'], intermediate_bytes=int(r0['line'][3]), final_bytes=int(r0['line'][4]), nodes=29,
        local_native=native['line'], reader_four_state=reader['line'], dispatch_four_state=dispatch['line'], dma_four_state=dma_gate['line'],
        compile_warnings_review=warnings_review(ideal_base),
        stall_profile='xorshift32 seed71; W acceptance and R creation gated by two RNG bits (nominal 75%); initial read and write-response delays 12 cycles instead of 1. One defined profile, not worst-case DDR.',
        scope='Complete diagnostic CNN frame; actual finite RAM arena/resident and ports; precompiled TB command scheduler; behavioral AXI DDR; target 150MHz only.',
        validation_limits=['Full frame and new combined core use native simulation, not whole-core four-state or gate-level simulation.',
            'Four-state reader/dispatcher and DMA tests are compositional local evidence.',
            'No production scheduler, real DDR PHY/refresh/multiple masters/CDC, full-frame fault recovery, real trained weights, or full baseline proof.',
            'Separate core and DMA mappings are not additive final XLR utilization; ADD and SRL also need physical packing.'],
        full_system_fit=False, final_sta=False, board_test=False, steady_real_frame_interval=False,
        evidence=evidence)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(status=result['status'], measurements=[{k:v for k,v in x.items() if k in ('profile','cycles','ms_at_target_150mhz','budget_pass','margin_cycles')} for x in measurements], core=core, full_system_fit=False)))


if __name__ == '__main__':
    main()
