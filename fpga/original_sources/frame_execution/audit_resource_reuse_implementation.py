"""Bind resource reuse results to exact RTL, simulator binaries and full-frame evidence."""
import argparse,collections,json,re
from pathlib import Path
from run_units import ROOT,sha
import audit_resource_reuse as structural

BASE=ROOT/'tmp/frame-execution'
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def records(folder,result):
    mf=folder/('native_inputs.json' if (folder/'native_inputs.json').exists() else 'inputs.json')
    assert sha(mf)==result['inputs_sha256'],mf
    manifest=read(mf)
    if 'snapshot' in manifest and 'source_manifest_sha256' in manifest:
        assert sha(manifest['binary'])==manifest['binary_sha256']
        source=Path(manifest['snapshot'])/'inputs.json'
        assert sha(source)==manifest['source_manifest_sha256']
        manifest=read(source)
    if 'binary' in manifest:assert sha(manifest['binary'])==manifest['binary_sha256']
    found={}
    for item in manifest['inputs']:
        path=Path(item['snapshot'])
        digest=item.get('sha256',item.get('snapshot_sha256'))
        assert digest and sha(path)==digest,path
        assert path.name not in found
        found[path.name]=digest
    return found,manifest

def map_case(name):
    folder=BASE/name;r=read(folder/'result.json')
    sources,_=records(folder,r)
    net=folder/'outflow/cnn_frame_datapath.map.v'
    report=folder/'outflow/cnn_frame_datapath.map.rpt'
    assert sha(net)==r['map_sha256'] and sha(report)==r['report_sha256']
    counts=dict(collections.Counter(re.findall(r'^\s*EFX_([A-Z0-9_]+)\s+\S+',net.read_text(),re.M)))
    assert counts==r['resources']
    for kind,number in counts.items():
        assert re.findall(r'^EFX_'+kind+r'\s*:\s*(\d+)\s*$',report.read_text(),re.M)==[str(number)]
    structural.CASE=folder
    hierarchy=structural.hierarchy()
    for key in ('LUT4','COMB4','ADD','SRL8','FF'):assert hierarchy[0]['inclusive'][key]==counts.get(key,0)
    partition=[]
    for row in hierarchy:
        if row['depth']==1 and row['path'].split('/')[-1]!='dut':
            partition.append(dict(name=row['path'].split('/')[-1],**row['inclusive']))
        elif row['depth']==2 and row['path'].split('/')[1]=='dut':
            partition.append(dict(name='/'.join(row['path'].split('/')[1:]),**row['inclusive']))
    for name0 in ('cnn_frame_datapath','cnn_frame_datapath/dut'):
        row=next(x for x in hierarchy if x['path']==name0)
        partition.append(dict(name=name0+'/own',**row['own']))
    for key in ('LUT4','COMB4','ADD','SRL8','FF'):
        assert sum(x[key] for x in partition)==counts.get(key,0)
    total=sum(counts.get(key,0) for key in ('LUT4','COMB4','ADD','SRL8'))
    assert counts['DSP24']==256 and counts['DSP48']==32
    assert counts['RAM10']+counts['DPRAM10']==120
    return dict(name=name,resources=counts,combination_primitive_sum=total,
        physical_dsp=160,ram=120,partition=partition,result_sha256=sha(folder/'result.json'),
        map_sha256=sha(net),sources=sources,physical_fit=False)

def gate(name):
    folder=BASE/name;r=read(folder/'result.json')
    assert 'PASS' in r['status']
    sources,_=records(folder,r)
    digest=r.get('log_sha256',r.get('simulation_sha256'))
    assert sha(folder/'simulation.log')==digest,name
    return dict(name=name,status=r['status'],line=r.get('line'),active_lines=r.get('active_lines'),
        result_sha256=sha(folder/'result.json'),sources=sources)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resources-only',action='store_true');a=p.parse_args()
    assert not a.output.exists()
    maps=[map_case(n) for n in ('byte-rotated-map1','reuse-comb4-map1','reuse-prefix-map1','reuse-integrated-map1')]
    final=maps[-1]
    assert [m['combination_primitive_sum'] for m in maps]==[82650,80378,77990,74666]
    gates=[gate(n) for n in ('reuse-credit8-unit1','reuse-prefix-unit1','reuse-route-dispatch1',
        'reuse-route-reader2','reuse-aligned-dma1','reuse-prefix-wrap3','reuse-integrated-unit1',
        'reuse-integrated-guard1','reuse-integrated-guard-fourstate1')]
    for n in ('reuse-integrated-unit1','reuse-integrated-guard1','reuse-integrated-guard-fourstate1'):
        g=next(x for x in gates if x['name']==n)
        common=set(final['sources']) & set(g['sources'])
        assert len(common)>=20
        assert all(final['sources'][key]==g['sources'][key] for key in common),n
        g['matching_synthesized_files']=len(common)
    desc=read(BASE/'reuse-descriptors1/result.json')
    prior=read(BASE/'production-descriptors3/result.json')
    assert desc['route_contract']['commands']==dict(compute=8789,dma=10440)
    assert desc['route_contract']['row_maps']==28 and desc['total_bytes']==729536
    assert [v['sha256'] for v in desc['streams'].values()]==[v['sha256'] for v in prior['streams'].values()]
    contract=ROOT/'network_ab/model_b_contract.json'
    assert sha(contract)==desc['network_contract_sha256']=='5750c4668d75077746ec5d4accf8773ab437540cf933c202f153a8797c21dd01'
    frames=[]
    if not a.resources_only:
        for name in ('reuse-integrated-frame1','reuse-integrated-profile1'):
            folder=BASE/name;r=read(folder/'result.json');sources,m=records(folder,r)
            assert r['status']=='CONTINUOUS_720P_RTL_NUMERIC_MEASUREMENT_PASS'
            assert sha(folder/'simulation.log')==r['log_sha256']
            assert r['commands']==19229 and r['line'][3:]==['65952000','2764800','430560','403200','6390','1fffffff']
            assert r['cycles']==sum(x['cycles'] for x in r['reasons'])
            assert abs(r['measured_frame_ms']-r['cycles']/150000)<1e-10
            assert r['full_baseline'] is False and r['physical_timing_pass'] is False
            assert all(sources.get(k)==v for k,v in final['sources'].items())
            frames.append(dict(name=name,cycles=r['cycles'],milliseconds=r['measured_frame_ms'],
                passes_assumed_150MHz_budget=r['cycles']<=5000000,margin_cycles=5000000-r['cycles'],
                reasons=r['reasons'],stalls=r['stalls'],dma_overlap=r['dma_overlap'],
                binary_sha256=m['binary_sha256'],matching_synthesized_files=len(final['sources']),
                result_sha256=sha(folder/'result.json')))
        assert frames[0]['binary_sha256']==frames[1]['binary_sha256']
    incomplete=[]
    for name in ('reuse-area2-map1','reuse-integrated-fourstate1'):
        folder=BASE/name
        if (folder/'result.json').exists():
            incomplete.append(dict(name=name,status=read(folder/'result.json')['status']))
        else:
            termination=folder/'termination.json'
            assert termination.exists(),str(folder)+' not closed'
            incomplete.append(dict(name=name,status=read(termination)['status'],record_sha256=sha(termination)))
    result=dict(status='RESOURCE_REUSE_MAP_AND_LOCAL_AUDIT_PASS' if a.resources_only else
        'RESOURCE_REUSE_SAME_SOURCE_FRAME_RESOURCE_AUDIT_PASS',maps=maps,local_gates=gates,frames=frames,
        saved_combination_primitives=maps[0]['combination_primitive_sum']-final['combination_primitive_sum'],
        saved_percent=100*(maps[0]['combination_primitive_sum']-final['combination_primitive_sum'])/maps[0]['combination_primitive_sum'],
        network_contract_sha256=sha(contract),descriptor_result_sha256=sha(BASE/'reuse-descriptors1/result.json'),
        incomplete=incomplete,resource_fit=False,full_baseline=False,production_scheduler=False,
        physical_timing_pass=False,board_test=False,
        limitations=['Combination primitive sum is structural comparison, not placed XLR utilization.',
        'Same-source full frame uses synthetic weights, finite TB commands, behavioral AXI memory and assumed 150MHz.',
        'Four-state local/guard tests do not mean complete full-frame four-state or postmap gate-level equivalence.',
        'Final camera/gesture/effect/display/Sapphire/four-master DDR and production command/ownership are not in this map.'])
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','saved_combination_primitives','saved_percent','resource_fit')}))
    for r in frames:print(json.dumps({k:r[k] for k in ('name','cycles','milliseconds','margin_cycles')}))
if __name__=='__main__':main()
