"""Bind full macro-floorplan proof while retaining all sensor73 release gates."""
from pathlib import Path
source=Path(__file__).with_name('package_video_sensor73.py');code=source.read_text()
old=" physical=Path(t['physical']);embedded(bindfile(physical/'physical_inputs.json'))"
new=old+"""
 from video_sensor73_macro_physical import verify_placed
 macro=verify_placed(physical);assert macro['count']==533 and macro['ordinary_anchors']==0
 anchor=bindfile(physical/'cnn_ram_placement.json');embedded(anchor)
 assert anchor['status']=='PRIOR_PASSED_MACRO_FLOORPLAN_TEST_NOT_RELEASE'
 assert anchor['compatibility']['ram_cells']==245 and anchor['compatibility']['dsp_cells']==288
 assert anchor['compatibility']['all_ram_modes_and_clock_connections_match']
 for name in ('video_sensor73_macro_physical.py','place_video_sensor73_macros.py','review_video_sensor73_macros.py'):
  bindfile(HERE/name)
"""
assert code.count(old)==1;code=code.replace(old,new)
old=" else:bindfile(HERE/'route_video_service.py',ri['runner_sha256'])"
new=""" elif 'route_clock_skew_comparison' in ri:
  override=ri['route_clock_skew_comparison']
  assert override==dict(value='off',placement_recipe='CONGESTION_2',cli_override_only=True,original_sdc_changed=False,source_or_placement_changed=False)
  bindfile(HERE/'route_video_service.py',ri['base_route_runner_sha256'])
  bindfile(HERE/'route_video_sensor73_no_skew.py',ri['runner_sha256'])
  bindfile(a.route/'runner.py',ri['runner_sha256'])
  assert 'beneficial_skew=off' in (physical/'route_service.cmd').read_text()
  bindfile(physical/'route_service.cmd')
  reuse=ri['placement_reuse'];origin=Path(reuse['source'])
  bindfile(origin/'physical_inputs.json',reuse['source_physical_sha256'])
  bindfile(origin/'before_route/result.json')
  assert reuse['copied_not_reexecuted'] and reuse['ordinary_locations_not_newly_locked']
  for name,h in reuse['geometry'].items():
   bindfile(origin/'outflow'/name,h);bindfile(physical/'outflow'/name,h)
 else:bindfile(HERE/'route_video_service.py',ri['runner_sha256'])"""
assert code.count(old)==1;code=code.replace(old,new)
old=" (out/'manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf8')"
new=""" m['physical_optimization_schedule']={'global_placement':'CONGESTION_2','routing':'CONGESTION_2','routing_beneficial_skew':ri.get('route_clock_skew_comparison',{}).get('value','on'),'hard_macro_anchors':533,'ordinary_cell_anchors':0,'prior_route_imported':False}
 m['macro_floorplan_evidence']={'path':str(physical/'cnn_ram_placement.json'),'sha256':sha(physical/'cnn_ram_placement.json')}
"""+old
assert code.count(old)==1;code=code.replace(old,new)
old=" for f in (Path(__file__),HERE/'video_sensor73_physical.py'"
new=" for f in (source,Path(__file__),HERE/'video_sensor73_physical.py'"
assert code.count(old)==1;code=code.replace(old,new)
exec(compile(code,str(source)+':complete-macro-floorplan','exec'),globals())
