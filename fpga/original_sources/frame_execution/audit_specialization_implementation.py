"""Bind specialization resources and functional gates to immutable source hashes."""
import argparse,json
from pathlib import Path
from run_units import ROOT,sha
from audit_resource_reuse_implementation import map_case,gate,records,read

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
p.add_argument('--final-map',default='spec-map1');p.add_argument('--frames',nargs='+',default=['spec-frame1','spec-profile1'])
p.add_argument('--resources-only',action='store_true');a=p.parse_args()
base=ROOT/'tmp/frame-execution';old=map_case('reuse-integrated-map1')
names=['spec-map1']
if (base/'spec-rq-map1/result.json').is_file():names.append('spec-rq-map1')
maps=[map_case(n) for n in names];final=next(m for m in maps if m['name']==a.final_map)
gates=[gate(n) for n in ('spec-dma1','spec-reader1','spec-unit1','spec-guard1',
    'spec-guard-fourstate1','spec-rejection1','spec-rq-equiv1')]
if a.final_map=='spec-rq-map1':
    gates += [gate(n) for n in ('spec-rq-unit1','spec-rq-guard1','spec-rq-guard-fourstate1')]
if (base/'spec-subset-fourstate1/result.json').exists():gates.append(gate('spec-subset-fourstate1'))
for g in gates:
    common=set(final['sources'])&set(g['sources'])
    g['matching_synthesized_files']=sum(final['sources'][k]==g['sources'][k] for k in common)
    g['different_files']=[k for k in common if final['sources'][k]!=g['sources'][k]]
    # The second candidate changes only the RQ proven equivalent in its
    # dedicated four-state test. Clearly expose transferred vs identical evidence.
    if g['name']=='spec-rq-equiv1' and a.final_map=='spec-map1':
        # This optional experiment is not part of the selected candidate.
        g['unselected_experiment']=True
    else:
        assert set(g['different_files'])<=({'shared_rq_oc8_cluster.sv'} if a.final_map=='spec-rq-map1' else set()),g['name']
    if a.final_map=='spec-rq-map1' and g['name'].startswith('spec-rq-'):
        assert not g['different_files'],g['name']
    if g['name'] in ('spec-rq-guard1','spec-rq-guard-fourstate1'):
        assert g['matching_synthesized_files']==26
incomplete=[]
timeout_folder=base/'spec-subset-fourstate1'
if not (timeout_folder/'result.json').exists():
    termination=read(timeout_folder/'termination.json')
    assert termination['status']=='INCOMPLETE_WALL_TIMEOUT_NOT_PASS'
    assert sha(timeout_folder/'simulation.log')==termination['log_sha256']
    assert sha(timeout_folder/'inputs.json')==termination['inputs_sha256']
    incomplete.append(dict(name=timeout_folder.name,**termination,
        record_sha256=sha(timeout_folder/'termination.json')))
contract=read(ROOT/'output/toolchain/resource_specialization_contract_2026-10-01.json')
desc=read(base/'spec-descriptors1/result.json');prior=read(base/'reuse-descriptors1/result.json')
assert contract['status']=='SPECIALIZED_GEOMETRY_CONTRACT_PASS'
assert contract['network_contract_sha256']==desc['network_contract_sha256']==sha(ROOT/'network_ab/model_b_contract.json')
assert contract['commands']==desc['route_contract']['commands']==dict(compute=8789,dma=10440)
assert [x['sha256'] for x in desc['streams'].values()]==[x['sha256'] for x in prior['streams'].values()]
frames=[]
if not a.resources_only:
    for name in a.frames:
        folder=base/name;r=read(folder/'result.json');sources,m=records(folder,r)
        assert r['status']=='CONTINUOUS_720P_RTL_NUMERIC_MEASUREMENT_PASS'
        assert sha(folder/'simulation.log')==r['log_sha256']
        assert r['commands']==19229 and r['line'][3:]==['65952000','2764800','430560','403200','6390','1fffffff']
        assert r['cycles']==sum(x['cycles'] for x in r['reasons'])
        assert abs(r['measured_frame_ms']-r['cycles']/150000)<1e-10
        assert not r['full_baseline'] and not r['physical_timing_pass'] and not r['steady_frame_interval_measured']
        assert all(sources.get(k)==v for k,v in final['sources'].items())
        prior_frame=read(base/('reuse-integrated-frame1' if r['profile']==0 else 'reuse-integrated-profile1')/'result.json')
        frames.append(dict(name=name,profile=r['profile'],cycles=r['cycles'],milliseconds=r['measured_frame_ms'],
            margin_cycles=5000000-r['cycles'],reasons=r['reasons'],stalls=r['stalls'],
            measured_150MHz_budget_pass=r['cycles']<=5000000,
            cycle_change_from_baseline=r['cycles']-prior_frame['cycles'],
            baseline_result_sha256=sha(base/('reuse-integrated-frame1' if r['profile']==0 else 'reuse-integrated-profile1')/'result.json'),
            matching_synthesized_files=len(final['sources']),binary_sha256=m['binary_sha256'],result_sha256=sha(folder/'result.json')))
    assert len({f['binary_sha256'] for f in frames})==1
    assert {f['profile'] for f in frames}=={0,1}
changed=[name for name in final['sources'] if final['sources'][name]!=old['sources'][name]]
bundle=None
if a.final_map=='spec-rq-map1':
    package=ROOT/'hw/efinity/prototypes/frame_execution/variants/specialized_integrated'
    metadata=read(package/'candidate.json')
    assert metadata['frozen_base']=='tmp/frame-execution/reuse-integrated-map1'
    assert set(metadata['override_files'])==set(changed)
    assert metadata['map_sha256']==final['map_sha256']
    assert metadata['resource_counts']==final['resources']
    assert metadata['network_contract_sha256']==contract['network_contract_sha256']
    for name in changed:assert sha(package/name)==final['sources'][name]
    bundle=dict(candidate_sha256=sha(package/'candidate.json'),override_files=metadata['override_files'])
result=dict(status='SPECIALIZATION_RESOURCE_AND_LOCAL_AUDIT_PASS' if a.resources_only else 'SPECIALIZATION_SAME_SOURCE_FRAME_RESOURCE_AUDIT_PASS',
    audit_script_sha256=sha(__file__),
    baseline=old,maps=maps,selected_map=a.final_map,changed_synthesized_files=changed,gates=gates,frames=frames,incomplete=incomplete,
    saved_lut=old['resources']['LUT4']-final['resources']['LUT4'],saved_ff=old['resources']['FF']-final['resources']['FF'],
    saved_combination_primitives=old['combination_primitive_sum']-final['combination_primitive_sum'],
    contract_result_sha256=sha(ROOT/'output/toolchain/resource_specialization_contract_2026-10-01.json'),
    descriptor_result_sha256=sha(base/'spec-descriptors1/result.json'),network_contract_sha256=contract['network_contract_sha256'],
    candidate_bundle=bundle,resource_fit=False,physical_timing_pass=False,board_download_allowed=False,full_baseline=False,
    limitations=['Resource counts are mapped logic, not placed XLR.',
        'Four-state 12-block cluster subset timed out after 900s with 9 BLOCK lines and no final PASS; complete 24-block and full-frame four-state replay remain open.',
        'Full frames use synthetic weights, finite testbench command issue and behavioral AXI memory at assumed 150MHz.',
        'Production command issue, final peripherals/four-master DDR, actual PHY bandwidth and final P&R/STA remain open.'])
assert not a.output.exists();a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:result[k] for k in ('status','selected_map','changed_synthesized_files','saved_lut','saved_ff','saved_combination_primitives','resource_fit')},ensure_ascii=False))
for f in frames:print(json.dumps({k:f[k] for k in ('name','cycles','milliseconds','margin_cycles')}))
