"""One C2 route on an exact copy of the reviewed C1 detailed geometry."""
from pathlib import Path
source=Path(__file__).with_name('route_video_service.py');code=source.read_text()
old=" assert optimization in ('CONGESTION_1','CONGESTION_2','CONGESTION_3')"
new=""" assert optimization=='CONGESTION_2'
 assert meta['placement_refinement']['optimization']=='CONGESTION_1'
 refinement=json.loads((physical/'refinement-result.json').read_text())
 assert refinement['status']=='DETAILED_PLACEMENT_FINISHED_NOT_RELEASE' and refinement['exit_code']==0
 assert refinement['inputs_sha256']==sha(physical/'refinement-inputs.json')
 reuse=meta['placement_reuse'];origin=Path(reuse['source'])
 assert sha(origin/'physical_inputs.json')==reuse['source_physical_sha256']
 for n,h in reuse['geometry'].items():assert sha(physical/'outflow'/n)==sha(origin/'outflow'/n)==h"""
assert code.count(old)==1;code=code.replace(old,new)
old='runner_sha256=sha(__file__),hardware_access=False,board_release=False)'
new="runner_sha256=sha(__file__),base_route_runner_sha256=sha(source),placement_reuse=reuse,placement_refinement_route=dict(refinement_inputs_sha256=sha(physical/'refinement-inputs.json'),refinement_result_sha256=sha(physical/'refinement-result.json'),optimization=optimization,detailed_optimization='CONGESTION_1'),hardware_access=False,board_release=False)"
assert code.count(old)==1;code=code.replace(old,new)
exec(compile(code,str(source)+':reviewed-C1-geometry-C2-route','exec'),globals())
