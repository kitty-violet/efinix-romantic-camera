"""Bind the refined-placement runner while retaining every sensor73 release gate."""
from pathlib import Path
source=Path(__file__).with_name('package_video_sensor73.py')
code=source.read_text()
old=" if 'routing_only_override' in ri:"
new=""" if 'placement_refinement_route' in ri:
  refinement=ri['placement_refinement_route'];assert refinement['optimization']=='CONGESTION_1'
  rinputs=bindfile(physical/'refinement-inputs.json',refinement['refinement_inputs_sha256']);embedded(rinputs)
  rresult=bindfile(physical/'refinement-result.json',refinement['refinement_result_sha256'])
  assert rresult['status']=='DETAILED_PLACEMENT_FINISHED_NOT_RELEASE' and rresult['exit_code']==0
  assert rresult['inputs_sha256']==refinement['refinement_inputs_sha256']
  embedded(bindfile(physical/'refinement-global-review.json'))
  bindfile(HERE/'route_video_sensor73_congestion.py',ri['runner_sha256'])
  bindfile(a.route/'runner.py',ri['runner_sha256'])
  bindfile(HERE/'route_video_service.py',ri['base_route_runner_sha256'])
  bindfile(source)
 elif 'routing_only_override' in ri:"""
assert code.count(old)==1;code=code.replace(old,new)
old=" (out/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf8')"
new=""" m['physical_optimization_schedule']={'global_placement':'CONGESTION_2','detailed_placement':'CONGESTION_1','routing':'CONGESTION_1','new_ordinary_location_locks':0}
 m['refinement_evidence']={'inputs':str(physical/'refinement-inputs.json'),'result':str(physical/'refinement-result.json'),'global_review':str(physical/'refinement-global-review.json')}
"""+old
assert code.count(old)==1;code=code.replace(old,new)
exec(compile(code,str(source)+':refined-placement-evidence','exec'),globals())
