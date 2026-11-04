"""Full original gates for the bounded original-SDC convergence candidate."""
from pathlib import Path
from control_command_repair import once
SOURCE=Path(__file__).with_name('finish_cnn_run_budget_release.py')
code=SOURCE.read_text(encoding='utf8')
code=code.replace('finish_cnn_run_budget_release','finish_cnn_run_convergence_release').replace('cnn_run_budget_c1_flow','cnn_run_budget_convergence_flow')
code=once(code,"dict(before='CONGESTION_2',after='CONGESTION_1')","dict(before='CONGESTION_2',after='CONGESTION_2')")
code=once(code," base_extra(board,batch,physical,serial,bind,verify)"," assert bind(physical/'route_review.json')['budget_seconds']==2700\n base_extra(board,batch,physical,serial,bind,verify)")
exec(compile(code,str(SOURCE)+':original-SDC-convergence-release','exec'),globals())
