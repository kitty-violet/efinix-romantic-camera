"""Route a separately reviewed CONGESTION_1 detailed placement, retaining all gates."""
from pathlib import Path
source=Path(__file__).with_name('route_video_service.py')
code=source.read_text()
old="optimization=meta['optimization_change']['value']\n assert optimization in ('CONGESTION_1','CONGESTION_2','CONGESTION_3')"
new="""optimization=meta['placement_refinement']['optimization']
 assert optimization=='CONGESTION_1'
 refinement=json.loads((physical/'refinement-result.json').read_text())
 assert refinement['status']=='DETAILED_PLACEMENT_FINISHED_NOT_RELEASE' and refinement['exit_code']==0
 assert refinement['inputs_sha256']==sha(physical/'refinement-inputs.json')"""
assert code.count(old)==1;code=code.replace(old,new)
old="assert recipes==[optimization],('Placement recipe differs from route recipe',recipes,optimization)"
new="assert recipes==['CONGESTION_2'],('Initial global-placement recipe changed',recipes)\n assert meta['placement_refinement']['optimization']==optimization"
assert code.count(old)==1;code=code.replace(old,new)
old='runner_sha256=sha(__file__),hardware_access=False,board_release=False)'
new="runner_sha256=sha(__file__),base_route_runner_sha256=sha(source),placement_refinement_route=dict(refinement_inputs_sha256=sha(physical/'refinement-inputs.json'),refinement_result_sha256=sha(physical/'refinement-result.json'),optimization=optimization),hardware_access=False,board_release=False)"
assert code.count(old)==1;code=code.replace(old,new)
exec(compile(code,str(source)+':actual-detailed-placement','exec'),globals())
