"""Read-only reachability audit for specializing the frozen B1 CNN datapath.
Observed program invariants are candidates, not proved production restrictions.
No RTL changes, no synthesis, no savings prediction.
"""
import argparse,collections,json
from pathlib import Path
from run_units import ROOT,sha
import audit_resource_reuse as resources

def hist(rows,indices):
    c=collections.Counter(tuple(row[i] for i in indices) for row in rows)
    return [dict(values=list(k),count=v) for k,v in sorted(c.items())]
def fields(word):return [(word>>(32*i))&0xffffffff for i in range(32)]
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists()
    case=ROOT/'tmp/frame-execution/reuse-integrated-map1'
    r=json.loads((case/'result.json').read_text())
    m=json.loads((case/'inputs.json').read_text())
    assert sha(case/'inputs.json')==r['inputs_sha256']
    for f in m['inputs']:assert sha(f['snapshot'])==f['sha256']
    vec=ROOT/'tmp/frame-execution/joint-reorder-vectors1'
    vm=json.loads((vec/'frame_manifest.json').read_text())
    assert sha(vec/'program.hex')==vm['files']['program.hex']
    program=[fields(int(x,16)) for x in (vec/'program.hex').read_text().splitlines()]
    compute=[f for f in program if f[0]==2]
    dma=[f for f in program if f[0]!=2]
    assert len(compute)==8789 and len(dma)==10440
    bynode=collections.defaultdict(list)
    for f in compute:bynode[f[1]].append(f)
    geometry=[2,3,4,5,6,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,28]
    layers=[]
    for nid,rows in sorted(bynode.items()):
        layers.append(dict(index=nid,name=vm['nodes'][nid]['id'],commands=len(rows),
            field_values={str(i):sorted(set(f[i] for f in rows)) for i in geometry}))
    fused=[f for f in compute if f[6]&16]
    patterns={}
    for label,rows in [('all',compute),('fused',fused)]:
        patterns[label]=dict(
            flags=hist(rows,[6]),group_pairs=hist(rows,[4,5]),masks=hist(rows,[28]),
            pixels_rows=hist(rows,[2,3]),source_widths=hist(rows,[8]),
            skip_delta=hist(rows,[23]) if label=='fused' else None,
            output_layouts=[dict(planes=list(p0),bases=list(b0),count=n) for (p0,b0),n in
                collections.Counter((tuple((f[13]>>(2*j))&3 for j in range(f[3])),
                    tuple(((f[14]|(f[15]<<32))>>(10*j))&1023 for j in range(f[3]))) for f in rows).items()],
            skip_layouts=[dict(planes=list(p0),bases=list(b0),count=n) for (p0,b0),n in
                collections.Counter((tuple((f[20]>>(2*j))&3 for j in range(f[3])),
                    tuple(((f[21]|(f[22]<<32))>>(10*j))&1023 for j in range(f[3]))) for f in rows).items()] if label=='fused' else None)
    dma_modes=[]
    for (store,rgb),n in sorted(collections.Counter((f[0],f[6]) for f in dma).items()):
        rows=[f for f in dma if (f[0],f[6])==(store,rgb)]
        dma_modes.append(dict(store=store,rgb=rgb,commands=n,groups=hist(rows,[3]),
            planes=hist(rows,[4]),bases=hist(rows,[5]),columns=hist(rows,[2]),
            alignment=sorted(collections.Counter(f[7]%16 for f in rows).items()),
            target_low_port_bits=sorted(set((((i>>({1:0,2:1,4:2}[f[3]]))&3)<<1)|(i&1 if f[3]>1 else 0)
                for f in rows for i in range(f[2]*f[3])))))
    # Each projected loop below counts RTL format alternatives, not LUTs saved.
    assert all(f[3]==(1 if f[6] else 4) for f in dma)
    def cyclic(rows,field):
        return all(((f[field]>>(2*j))&3)==((f[field]&3)+j)%3
            for f in rows for j in range(f[3]))
    invariants=dict(output_planes_cyclic=cyclic(compute,13),
        fused_skip_planes_cyclic=cyclic(fused,20),
        fused_groups_c32_only=all(f[4]==4 and f[5]==4 for f in fused),
        fused_delta_zero_or_two_only=all(f[23] in (0,2) for f in fused),
        fields_constant_per_layer=[i for i in geometry if all(
            len({f[i] for f in rows})==1 for rows in bynode.values())],
        output_layout_count=len(patterns['all']['output_layouts']),
        fused_output_layout_count=len(patterns['fused']['output_layouts']))
    assert invariants['output_planes_cyclic'] and invariants['fused_skip_planes_cyclic']
    assert invariants['fused_groups_c32_only'] and invariants['fused_delta_zero_or_two_only']
    resources.CASE=case
    hierarchy=resources.hierarchy()
    modules={}
    for module in ('shared_rq_oc8_cluster','requantize_i32_mul18_q17_pipeline',
        'int8_dual_mac_lane_prefix_stream','shared_mac_stream_front','banked_result_fifo'):
        rows=[x for x in hierarchy if x['module'].split('(')[0]==module]
        modules[module]=dict(instances=len(rows),
            own={k:sum(x['own'][k] for x in rows) for k in hierarchy[0]['own']},
            inclusive={k:sum(x['inclusive'][k] for x in rows) for k in hierarchy[0]['own']})
    counter_cells=collections.defaultdict(lambda:collections.Counter())
    import re
    net=case/'outflow/cnn_frame_datapath.map.v'
    assert sha(net)==r['map_sha256']
    for kind,name in re.findall(r'^\s*EFX_([A-Z0-9_]+)\s+(\S+)\s+\(',net.read_text(),re.M):
        if any(x in name for x in ('wait_input_cycles','wait_pool_cycles','read_words','paired_words','cycles[')):
            counter_cells[kind][name.split('/')[0]]+=1
    result=dict(status='FROZEN_PROGRAM_SPECIALIZATION_ANALYSIS_PASS_NOT_SYNTHESIS',
        case=str(case),map_sha256=sha(net),input_manifest_sha256=sha(case/'inputs.json'),
        network_contract_sha256=sha(ROOT/'network_ab/model_b_contract.json'),
        program_sha256=sha(vec/'program.hex'),compute_commands=len(compute),dma_commands=len(dma),
        layers=layers,compute_patterns=patterns,dma_modes=dma_modes,invariants=invariants,module_resources=modules,
        diagnostic_named_cells={k:dict(v) for k,v in counter_cells.items()},
        caveats=['Zero-use mode is observed in one complete fixed B1 program, not all future programs.',
        'Layer geometry may be specialized; trained weights, bias, scales and residual alignment stay runtime.',
        'Hierarchy counts are mapped attributions and cannot be called entirely removable.',
        'No functional RTL, synthesis, P&R or simulation changed by this analysis.'])
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','compute_commands','dma_commands','invariants')},ensure_ascii=False))
    print(json.dumps(dict(dma_modes=[{k:v for k,v in mode.items() if k not in ('bases',)} for mode in dma_modes],
        compute_patterns={name:{k:v for k,v in data.items() if k not in ('output_layouts','skip_layouts')}
            for name,data in patterns.items()}),ensure_ascii=False))
if __name__=='__main__':main()
