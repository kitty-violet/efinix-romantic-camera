"""One documented beneficial-skew OFF comparison on identical macro geometry.

This changes the implementation algorithm only. Original clocks, SDC and all
setup/hold/CDC/PLL release gates remain required; no board access is performed.
"""
from pathlib import Path
source=Path(__file__).with_name('route_video_service.py');code=source.read_text()
old=" assert optimization in ('CONGESTION_1','CONGESTION_2','CONGESTION_3')"
new=""" assert optimization=='CONGESTION_2'
 assert meta['route_clock_skew_comparison']==dict(value='off',placement_recipe='CONGESTION_2',cli_override_only=True,original_sdc_changed=False,source_or_placement_changed=False)
 reuse=meta['placement_reuse'];origin=Path(reuse['source'])
 assert sha(origin/'physical_inputs.json')==reuse['source_physical_sha256']
 for n,h in reuse['geometry'].items():assert sha(physical/'outflow'/n)==sha(origin/'outflow'/n)==h"""
assert code.count(old)==1;code=code.replace(old,new)
old="optimization_level='+optimization+' --timeout '"
new="optimization_level='+optimization+' beneficial_skew=off --timeout '"
assert code.count(old)==1;code=code.replace(old,new)
old='runner_sha256=sha(__file__),hardware_access=False,board_release=False)'
new="runner_sha256=sha(__file__),base_route_runner_sha256=sha(source),placement_reuse=reuse,route_clock_skew_comparison=meta['route_clock_skew_comparison'],hardware_access=False,board_release=False)"
assert code.count(old)==1;code=code.replace(old,new)
exec(compile(code,str(source)+':same-macros-clock-skew-off','exec'),globals())
