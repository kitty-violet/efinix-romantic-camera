"""Bind sparse RCF inputs in addition to every original functional/physical gate."""
from pathlib import Path
from control_command_repair import once
SOURCE=Path(__file__).with_name('finish_cnn_run_budget_release.py')
code=SOURCE.read_text(encoding='utf8')
code=code.replace('finish_cnn_run_budget_release','finish_cnn_run_preserved_release')
code=code.replace('cnn_run_budget_c1_flow','cnn_run_budget_preserved_flow')
code=code.replace("HERE/'cnn_run_budget_place.py'","HERE/'cnn_run_budget_preserved_place.py'")
code=once(code,"dict(before='CONGESTION_2',after='CONGESTION_1')","dict(before='CONGESTION_2',after='CONGESTION_2')")
code=once(code," for n in ('arena_island_placement.json'", """ from cnn_run_budget_preserved_locations import verify_placed as verify_preserved
 assert verify_preserved(physical)['nets']==82
 assert bind(physical/'route_review.json')['budget_seconds']==2700
 proof=bind(physical/'preserved_route_placement.json');verify(proof['evidence'])
 verify({str(physical/n):h for n,h in proof['generated_files'].items()})
 assert bind(physical/'physical_inputs.json')['preserved_route_placement_sha256']==sha(physical/'preserved_route_placement.json')
 bind(physical/'preserved_route_actual_locations.json')
 for n in ('cnn_run_budget_preserved_locations.py','cnn_run_budget_preserved_place.py','cnn_run_budget_preserved_flow.py','finish_cnn_run_budget_release.py'):
  verify({str(HERE/n):sha(HERE/n)})
 for n in ('arena_island_placement.json'""")
exec(compile(code,str(SOURCE)+':sparse-RCF-release','exec'),globals())
