"""Freeze an explicitly reviewed placement for one bounded TIMING_1 route."""
import argparse,json,shutil
from pathlib import Path
from run_units import ROOT,HERE,sha
from rebind_native_placement import translate
from local_driver_board import audit_delta

def main():
    p=argparse.ArgumentParser();p.add_argument('--review',type=Path,required=True);a=p.parse_args()
    review=json.loads(a.review.read_text());assert review['decision']=='PROCEED_TO_BOUNDED_ROUTE'
    source=Path(review['placement']).resolve();board=Path(review['board']).resolve();shorts=Path(review['shorts']).resolve()
    root=Path(review['physical']).resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    assert review['rationale'] and review['remaining_risks'] and review['budget_seconds']==1350
    assert review['route_profile_cli_override']=={'before':'CONGESTION_2','after':'TIMING_1'}
    for f,h in review['reviewed_evidence'].items():assert sha(f)==h,f
    audit_delta(board);sr=json.loads((shorts/'result.json').read_text())
    assert sr['status']=='LOCAL_DRIVER_SAME_MAP_SHORT_INTEGRATION_PASS'
    assert sr['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v')
    for f,h in sr['evidence'].items():assert sha(f)==h,f
    result=json.loads((source/'result.json').read_text());assert result['status']=='PLACEMENT_ONLY_FINISHED_NOT_RELEASE'
    assert not (source/'outflow/ti60f225_oob.troutingtraces').exists()
    meta=json.loads((source/'physical_inputs.json').read_text());assert Path(meta['base'])==board
    for row in meta['inputs']:assert sha(row['snapshot'])==row['sha256']
    sdc,pins=translate((source/'ti60f225_oob.sdc').read_text(),(source/'outflow/ti60f225_oob.map.v').read_text(),(source/'outflow/ti60f225_oob.out.placeloc').read_text())
    original={str(source/n):sha(source/n) for n in ('physical_inputs.json','result.json','ti60f225_oob.sdc','placement_review_inputs.json')}
    geometry={n:sha(source/n) for n in ('outflow/ti60f225_oob.map.v','outflow/ti60f225_oob.vdb','outflow/ti60f225_oob.place',
        'outflow/ti60f225_oob.out.placeloc','work_pnr/ti60f225_oob.net_proto','fixed_dsp.placeloc','ti60f225_oob.xml','ti60f225_oob.peri.xml','efx_pnr_settings.ini')}
    shutil.copytree(source,root);(root/'ti60f225_oob.sdc').write_text(sdc,encoding='utf8')
    for row in meta['inputs']:
        old=Path(row['snapshot']);row['snapshot']=str(root/old.relative_to(source));row['sha256']=sha(row['snapshot'])
    rebound=dict(status='EXACT_GRAY_ENDPOINT_NAME_REBINDING_NOT_TIMING_PASS',source=str(source),original=original,geometry=geometry,pins=pins,
        script_sha256=sha(__file__),translator_sha256=sha(HERE/'rebind_native_placement.py'),buses=10,destination_bits=70,
        timing_values_unchanged=True,clock_relationships_unchanged=True,placement_rerun=False,hardware_access=False,board_release=False)
    (root/'endpoint_rebinding.json').write_text(json.dumps(rebound,indent=2))
    meta.update(placement_origin=str(source),endpoint_rebinding_sha256=sha(root/'endpoint_rebinding.json'))
    meta['clock_profile']['before_endpoint_rebinding_sha256']=meta['clock_profile']['after_sha256']
    meta['clock_profile']['after_sha256']=sha(root/'ti60f225_oob.sdc')
    (root/'physical_inputs.json').write_text(json.dumps(meta,indent=2))
    for name,h in geometry.items():assert sha(root/name)==h
    for suffix in ('-local-driver-wrapper','-place-wrapper','-place-driver'):
        for ext in ('.py','.json'):shutil.copy2(source.parent/(source.name+suffix+ext),root.parent/(root.name+suffix+ext))
    review['placement_review_sha256']=sha(root/'placement_review_inputs.json')
    evidence=dict(review['reviewed_evidence']);evidence.update(sr['evidence']);evidence.update(original)
    for f in (root/'physical_inputs.json',root/'endpoint_rebinding.json',shorts/'result.json',
        HERE/'route_local_driver_board.py',HERE/'prepare_local_driver_route.py',a.review):evidence[str(f)]=sha(f)
    review['required_evidence']=evidence
    (root/'route_review.json').write_text(json.dumps(review,indent=2));print('REVIEWED_LOCAL_DRIVER_ROUTE_PREPARED')

if __name__=='__main__':main()
