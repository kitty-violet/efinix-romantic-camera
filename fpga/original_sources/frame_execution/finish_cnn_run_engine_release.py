"""Retain every finite-run release gate for the engine-placement candidate."""
from pathlib import Path
from control_command_repair import once
SOURCE=Path(__file__).with_name('finish_cnn_run_budget_release.py')
code=SOURCE.read_text(encoding='utf8')
code=code.replace('finish_cnn_run_budget_release','finish_cnn_run_engine_release')
code=code.replace('cnn_run_budget_c1_flow','cnn_run_budget_engine_flow')
code=code.replace("HERE/'cnn_run_budget_place.py'","HERE/'cnn_run_budget_engine_place.py'")
code=once(code,"dict(before='CONGESTION_2',after='CONGESTION_1')","dict(before='CONGESTION_2',after='CONGESTION_2')")
code=once(code," for n in ('arena_island_placement.json'", " from cnn_run_budget_engine_locations import verify_placed as verify_engine\n assert verify_engine(physical)['count']==9084\n for n in ('engine_island_placement.json','engine_island_actual_locations.json'):\n  r=bind(physical/n)\n  if 'evidence' in r:verify(r['evidence'])\n verify({str(HERE/'cnn_run_budget_engine_locations.py'):sha(HERE/'cnn_run_budget_engine_locations.py'),str(HERE/'cnn_run_budget_engine_place.py'):sha(HERE/'cnn_run_budget_engine_place.py'),str(HERE/'finish_cnn_run_budget_release.py'):sha(SOURCE)})\n for n in ('arena_island_placement.json'")
exec(compile(code,str(SOURCE)+':engine-layout-release','exec'),globals())
