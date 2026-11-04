"""Every finite-run release gate plus full connected-cone placement lineage."""
from pathlib import Path
from control_command_repair import once
SOURCE=Path(__file__).with_name('finish_cnn_run_budget_release.py')
code=SOURCE.read_text(encoding='utf8')
code=code.replace('finish_cnn_run_budget_release','finish_cnn_run_cone_release')
code=code.replace('cnn_run_budget_c1_flow','cnn_run_budget_cone_flow')
code=code.replace("HERE/'cnn_run_budget_place.py'","HERE/'cnn_run_budget_cone_place.py'")
code=once(code,"dict(before='CONGESTION_2',after='CONGESTION_1')","dict(before='CONGESTION_2',after='CONGESTION_2')")
code=once(code," for n in ('arena_island_placement.json'", " from cnn_run_budget_cone_locations import verify_placed as verify_cone\n assert verify_cone(physical)['count']==27572\n for n in ('cnn_cone_placement.json','cnn_cone_actual_locations.json'):\n  r=bind(physical/n)\n  if 'evidence' in r:verify(r['evidence'])\n for n in ('cnn_run_budget_cone_locations.py','cnn_run_budget_cone_place.py','cnn_run_budget_cone_flow.py','cnn_run_budget_engine_locations.py','cnn_run_budget_engine_place.py','cnn_run_budget_engine_flow.py','finish_cnn_run_budget_release.py'):\n  verify({str(HERE/n):sha(HERE/n)})\n for n in ('arena_island_placement.json'")
exec(compile(code,str(SOURCE)+':connected-cone-release','exec'),globals())
