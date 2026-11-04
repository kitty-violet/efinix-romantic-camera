"""Same bounded C2 route and complete gates for connected CNN data cones."""
from pathlib import Path
from control_command_repair import once
SOURCE=Path(__file__).with_name('cnn_run_budget_engine_flow.py')
code=SOURCE.read_text(encoding='utf8').replace('cnn_run_budget_engine_locations','cnn_run_budget_cone_locations')
code=once(code,"['count']==9084","['count']==27572")
exec(compile(code,str(SOURCE)+':cnn-connected-cone-flow','exec'),globals())
